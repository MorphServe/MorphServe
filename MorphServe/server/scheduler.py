from collections import deque
import time

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.utils import cdiv
from swiftllm.server.structs import Request
from swiftllm.server.memory_utils import org2quant_and_update_kvcache, quant2org_and_update_kvcache


class RequestIdManager:
    """
    A class that maintains available request ids
    """
    def __init__(self, max_id: int):
        # Id should be in range [0, max_id)
        # Question: How to make the IDs always available, even there are more than max_id has been served? 
        # Answer: the available IDs are dynamicly appending into the available_ids list if the req is finished.
        self.max_id = max_id
        self.available_ids = list(range(max_id))
        self.available_ids.reverse()    # This reverse is not necessary. We have
                                        # this for more convinent debugging since
                                        # we always pop from the end and after
                                        # reversing, smaller ids are popped first
    
    def get_id(self) -> int:
        if not self.available_ids:
            raise RuntimeError("No more available request ids. Please try to increase `max_seqs_in_block_table`")
        return self.available_ids.pop()
    
    def free_id(self, req_id: int):
        self.available_ids.append(req_id)
    
    def free_ids(self, req_ids: list[int]):
        self.available_ids.extend(req_ids)


class Scheduler:
    """
    A strict FCFS scheduler for the LLM engine, which supports paged attention
    as well as swapping in/out
    """

    def __init__(self, model_config: LlamaModelConfig, engine_config: EngineConfig, num_gpu_blocks: int):
        self.model_config = model_config
        self.engine_config = engine_config
        self.num_gpu_blocks = num_gpu_blocks

        # Request in the following three deques are sorted by their arrival time
        self.waiting_q = deque()
        self.running_q: list[Request] = []
        # Question: Is this swapped_q containing swapped in or swapped out?   
        # Answer: it is holding swapped out sequence, and aiming to swap in during the get_next_batch()
        self.swapped_q = deque()

        # Number of GPU blocks occupied by decoding requests
        # This number should always equal to sum(self._get_block_needed(req) for req in self.running_q)
        self.num_decoding_gpu_blocks = 0
        self.num_free_cpu_blocks = engine_config.num_cpu_blocks

        self.request_id_manager = RequestIdManager(engine_config.max_seqs_in_block_table)
    
    def _get_block_needed(self, request: Request) -> int:
        """
        Get the number of blocks needed for a request
        """
        return cdiv(request.prompt_len + request.get_cur_output_len(), self.engine_config.block_size)
    
    def on_requests_arrival(self, requests: list[Request]):
        """
        Called when a batch of new requests arrives and finishes tokenization
        """
        self.waiting_q.extend(requests)
    
    def try_quant2org(self, engine):
        """
        Check if the new kv cache blocks can be freed and the remaining kv cache blocks can still accommodate the new request
        """
        # When the following conditions are met, we can do quant2org:
        # 1. There are quantized layers
        # 2. There are no waiting requests
        # 3. There are enough free blocks even after the quant2org
        if len(engine.model.layer_quant_list) > 0 and not self.waiting_q and engine.model.gpu_block_manager.num_free_blocks >= engine.model.kv_cache_new_block_size * engine.engine_config.num_layers_to_quantize:
            # Quant2Org at max `num_layers_to_quantize` times per schedule.
            for i in range(engine.engine_config.num_layers_to_quantize):
                if len(engine.model.layer_quant_list) > 0 and engine.model.gpu_block_manager.num_free_blocks > engine.model.kv_cache_new_block_size * (engine.engine_config.num_layers_to_quantize // 2):
                    quant_finished = quant2org_and_update_kvcache(engine)
                    if not quant_finished:
                        break

    def get_next_batch(self, engine) -> tuple[list[Request], list[Request], list[Request]]:
        """
        Called when the engine wants a new batch to be forwarded
        Returns (new_batch, newly_swapped_in, newly_swapped_out)
        """
        # Org2Quant at max `num_layers_to_quantize` times per schedule.
        layer_quantized_num_left = engine.engine_config.num_layers_to_quantize
        if not self.swapped_q:
            # Try to launch a new prefill batch
            cur_batch = []
            cur_batch_block_needed = 0
            cur_num_tokens_sum = 0
            while self.waiting_q:
                cur_seq: Request = self.waiting_q[0]
                # How many KV cache blocks needed to perform the inference for this req.
                # Round up to the nearest integer of the blocks needed 
                cur_seq_block_needed = self._get_block_needed(cur_seq)

                if engine.engine_config.quant_serve:
                    # New(layer replacement) Check if the current batch can still accommodate the new request
                    if  len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                        len(self.running_q)+len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                        cur_num_tokens_sum + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
                        layer_replacement_needed = (cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks - self.num_gpu_blocks + engine.model.kv_cache_new_block_size - 1) // engine.model.kv_cache_new_block_size
                        if layer_replacement_needed <= 0:
                            cur_batch.append(cur_seq)
                            cur_batch_block_needed += cur_seq_block_needed
                            cur_num_tokens_sum += cur_seq.prompt_len
                            self.waiting_q.popleft()
                        elif engine.model.next_unquantized_layer >= engine.engine_config.num_layers_to_quantize_max and layer_replacement_needed <= layer_quantized_num_left and (engine.model.next_unquantized_layer-layer_replacement_needed) >= 0: # when the next unquantized layer is 0, and the layer_replacement_needed is 1, it is still valid;
                            # print(f"\n[Scheduler: Org2Quant] (cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks - self.num_gpu_blocks) // engine.model.kv_cache_new_block_size: {cur_batch_block_needed} + {cur_seq_block_needed} + {self.num_decoding_gpu_blocks} - {self.num_gpu_blocks} // {engine.model.kv_cache_new_block_size}, layer_replacement_needed: {layer_replacement_needed}, layer_quantized_num_left: {layer_quantized_num_left}, engine.model.next_unquantized_layer: {engine.model.next_unquantized_layer}")
                            
                            # For moderate layer replacement, it is better to preserve the model accuracy
                            # for i in range(layer_replacement_needed):  
                            #     org2quant_and_update_kvcache(engine)
                            #  layer_quantized_num_left -= layer_replacement_needed

                            # For aggressive layer replacement, it is better to do layer replacement as soon as possible
                            for i in range(layer_quantized_num_left):
                                org2quant_and_update_kvcache(engine)
                            layer_quantized_num_left = 0

                            cur_batch.append(cur_seq)
                            cur_batch_block_needed += cur_seq_block_needed
                            cur_num_tokens_sum += cur_seq.prompt_len
                            self.waiting_q.popleft()
                        else:
                            # print(f"[Scheduler: Org2Quant not enough] (cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks - self.num_gpu_blocks) // engine.model.kv_cache_new_block_size: {cur_batch_block_needed} + {cur_seq_block_needed} + {self.num_decoding_gpu_blocks} - {self.num_gpu_blocks} // {engine.model.kv_cache_new_block_size}, layer_replacement_needed: {layer_replacement_needed}, layer_quantized_num_left: {layer_quantized_num_left}, engine.model.next_unquantized_layer: {engine.model.next_unquantized_layer}")
                            # print(f"It just mean: the current layer quantization num is not enough, or the next unquantized layer is not enough. If so, next line `exit()` should be removed.")
                            # Just for double check, if everything is correct, the exit() should be removed
                            # It just mean: the current layer quantization num is not enough, or the next unquantized layer is not enough
                            # exit()
                            break
                    else:
                        # if len(cur_batch)+1 > self.engine_config.max_batch_size:
                        #     print(f"[Scheduler] cur_batch+1 > self.engine_config.max_batch_size: {len(cur_batch)+1} > {self.engine_config.max_batch_size}")
                        # if len(self.running_q)+len(cur_batch)+1 > self.engine_config.max_batch_size:
                        #     print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[Scheduler] len(self.running_q)+len(cur_batch)+1 > self.engine_config.max_batch_size: {len(self.running_q)+len(cur_batch)+1} > {self.engine_config.max_batch_size}")
                        # if cur_num_tokens_sum + cur_seq.prompt_len > self.engine_config.max_tokens_in_batch:
                            # print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[Scheduler] cur_num_tokens_sum + cur_seq.prompt_len > self.engine_config.max_tokens_in_batch: {cur_num_tokens_sum + cur_seq.prompt_len} > {self.engine_config.max_tokens_in_batch}")
                        break
                else:
                    # Original Check if the current batch can still accommodate the new request
                    if  len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                        len(self.running_q)+len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                        cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks <= self.num_gpu_blocks and \
                        cur_num_tokens_sum + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
                        cur_batch.append(cur_seq)
                        cur_batch_block_needed += cur_seq_block_needed
                        cur_num_tokens_sum += cur_seq.prompt_len
                        self.waiting_q.popleft()
                    else:
                        break

            if cur_batch:
                # Going to launch a prefill batch
                # If you want decoding requests to be piggybacked, you can do it here
                queued_time = time.perf_counter()  # Record queued time
                for req in cur_batch:
                    req.request_id = self.request_id_manager.get_id()
                    req.req_metrics.queued_time = queued_time
                self.running_q.extend(cur_batch)
                self.num_decoding_gpu_blocks += cur_batch_block_needed
                if engine.engine_config.quant_serve and layer_quantized_num_left == engine.engine_config.num_layers_to_quantize:
                    self.try_quant2org(engine)
                return cur_batch, [], []
        
        # Try to launch a decoding batch
        # TODO Optimize this `sum` if possible
        self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.running_q)
        newly_swapped_out = []
        while len(self.running_q) > self.engine_config.max_batch_size or \
              self.num_decoding_gpu_blocks > self.num_gpu_blocks:
            # len(self.running_q) > self.engine_config.max_batch_size should not happen, because we have already checked it in the get_next_batch()
            if len(self.running_q) > self.engine_config.max_batch_size:
                # print(f"[Scheduler] len(self.running_q) > self.engine_config.max_batch_size: {len(self.running_q)} > {self.engine_config.max_batch_size}")
                exit()

            if engine.engine_config.quant_serve:
                # New(layer replacement) policy: do layer replacement when the kv cache is full to increase the KV blocks available, 
                # Until the self.num_decoding_gpu_blocks <= self.num_gpu_blocks. 
                # or the next unquantized layer is 0, and the layer_quantized_num_left is 0.
                if engine.model.next_unquantized_layer >= 0 and layer_quantized_num_left > 0:
                    # print(f"[Scheduler_Swap_Out] Org2Quant: self.num_decoding_gpu_blocks > self.num_gpu_blocks: {self.num_decoding_gpu_blocks} > {self.num_gpu_blocks}")
                    org2quant_and_update_kvcache(engine)
                    layer_quantized_num_left -= 1
                    # print(f"[Scheduler_Swap_Out] After Org2Quant: self.num_decoding_gpu_blocks: {self.num_decoding_gpu_blocks}, self.num_gpu_blocks: {self.num_gpu_blocks}, engine.model.next_unquantized_layer: {engine.model.next_unquantized_layer}, layer_quantized_num_left: {layer_quantized_num_left}")
                else:
                    # This should not happen, because we have already replaced the layers = 4, so that the KV cache should be enough.
                    # print(f"[Scheduler_Swap_Out] ELSE: self.num_decoding_gpu_blocks: {self.num_decoding_gpu_blocks}, self.num_gpu_blocks: {self.num_gpu_blocks}, engine.model.next_unquantized_layer: {engine.model.next_unquantized_layer}, layer_quantized_num_left: {layer_quantized_num_left}")
                    # exit()
                    victim = self.running_q.pop()
                    self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
                    newly_swapped_out.append(victim)
            else:   
                # Original policy: swap out the last running seq when the kv cache is full
                victim = self.running_q.pop()
                self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
                newly_swapped_out.append(victim)

        newly_swapped_out.reverse()   # Keep it in the order of arrival time

        newly_swapped_in = []
        if newly_swapped_out:
            self.swapped_q.extendleft(newly_swapped_out)
        else:
            # No swap-out triggered, try to swap in some requests if possible
            while self.swapped_q:
                cur_seq = self.swapped_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if len(self.running_q) + 1 <= self.engine_config.max_batch_size and \
                   self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
                    self.running_q.append(cur_seq)
                    self.num_decoding_gpu_blocks += num_cur_seq_blocks
                    self.swapped_q.popleft()
                    newly_swapped_in.append(cur_seq)
                    # If there are blocks swapped in, We do not do quant2org this round, 
                    # because there might some free blocks would be used by the swapped in requests
                    layer_quantized_num_left = -1
                else:
                    break
        if engine.engine_config.quant_serve and layer_quantized_num_left == engine.engine_config.num_layers_to_quantize:
            self.try_quant2org(engine)
        return self.running_q, newly_swapped_in, newly_swapped_out
    
    def on_batch_finish(self, batch: list[Request]):
        """
        Called when a batch finishes
        """
        self.request_id_manager.free_ids([
            req.request_id
            for req in batch
            if req.is_finished()
        ])
        self.running_q = [
            req
            for req in self.running_q
            if not req.is_finished()
        ]
