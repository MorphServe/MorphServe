import asyncio
import functools
from typing import AsyncGenerator
import time
import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB

from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput, ServerMetrics
from .scheduler import Scheduler

from .quantized_layer_manager import AWQLayerManager
from swiftllm.server.memory_utils import initialize_memory_tracking, org2quant_and_update_kvcache, quant2org_and_update_kvcache


class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.org_model_path)
        self.initialized = False
        
        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []

        # Metrics tracking
        self.server_metrics = ServerMetrics()
        
        # Original model on CPU
        self.original_model_cpu = None

        # Quantized model on GPU
        self.awq_layer_manager = None

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        # an event loop is the engine that runs async tasks, handles await, schedules coroutines, and coordinates I/O events
        self.event_loop = asyncio.get_event_loop()
        
        print("[Engine] Initializing original model on GPU...")
        self.model = LlamaModel(self.engine_config, self.engine_config.org_model_path)
        print("[Engine] Loading original model weights on GPU...")
        self.model.load_weights()
        self.model.eval()
        print("[Engine] Initializing layer memory tracking for original model on GPU...")
        initialize_memory_tracking(self.model, device_on_gpu=True)
        print(f"[Engine] self.model.org_layer_param_size: {self.model.org_layer_param_size}")

        print("[Engine] Initializing original model on CPU...")
        self.original_model_cpu = LlamaModel(self.engine_config, self.engine_config.org_model_path)
        print("[Engine] Loading original model weights on CPU...")
        self.original_model_cpu.load_weights(load_to_cpu=True)
        self.original_model_cpu.eval()
        print("[Engine] Initializing layer memory tracking for original model on CPU...")
        initialize_memory_tracking(self.original_model_cpu, device_on_gpu=False)

        print(f"[Engine] Loading quantized model...")
        self.awq_layer_manager = AWQLayerManager(
            self.engine_config.quantized_model_path,
            self.model_config,
            self.engine_config
        )

        # for i in range(self.engine_config.num_layers_to_quantize):
        #     org2quant_and_update_kvcache(self, update_kvcache=False)

        # Todo: Need to double check that if using the original model performing the profiling is correct.
        #       If not, need to use the quantized model to perform the profiling,
        #       because the peak memory usage is higher from the quantized model compared with the original model,
        #       due to the dequantization process.
        print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")
        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)
        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.engine_config, num_gpu_blocks)
        print("[Engine] Initializing tokenization engine...")
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)
        print(f"[Engine] Prewarming the layer replacement...")
        
        # for i in range(self.engine_config.num_layers_to_quantize):
        #     quant2org_and_update_kvcache(self, update_kvcache=False)

        for i in range(self.engine_config.num_layers_to_quantize):
            org2quant_and_update_kvcache(self)
        for i in range(self.engine_config.num_layers_to_quantize):
            quant2org_and_update_kvcache(self)

        print("[Engine] Model initialized")
        self.initialized = True

        # exit()

        # restore_layer_in_place(self, layer_id=layer_id)
        # print(f"[Engine] model.layer_quant_list: {self.model.layer_quant_list}")
        # print(f"[Engine] model.is_layer_quant_list: {self.model.is_layer_quant_list}")
        # print(f"[Engine] model.next_unquantized_layer: {self.model.next_unquantized_layer}")

        if self.engine_config.awq_mode:
            assert not self.engine_config.quant_serve, "AWQ mode must be used with quant_serve=False"
            if self.engine_config.awq_half_mode:
                quant_layer_total = self.model_config.num_layers // 2
            else:
                quant_layer_total = self.model_config.num_layers
            print("~"*50, f"Test the quantizing {quant_layer_total} layers", "~"*50)
            for i in range(quant_layer_total):
                org2quant_and_update_kvcache(self)

        # Test the quantizing layer and restore the original layer
        # print('\n',"~"*50, "Test the quantizing layer and restore the original layer", "~"*50)
        # print(f"[Engine] model.transformer_layers length: {len(self.model.transformer_layers)}")
        # print(f"[Engine] model.transformer_layers[0]: {self.model.transformer_layers[0]}")
        # print(f"[Engine] model.transformer_layers[0] vars: {vars(self.model.transformer_layers[0])}")
        # print(f"[Engine] model.transformer_layers[0].weight vars: {vars(self.model.transformer_layers[0].weight)}")
        
        # for layer_id in range(self.engine_config.num_layers_to_quantize):
        #     print(f"\n[Engine] Quantizing layer {layer_id}...")
        #     org2quant_and_update_kvcache(self)
        # print(f"[Engine] After quantizing layer {layer_id}:")
        # print(f"[Engine] model.transformer_layers length: {len(self.model.transformer_layers)}")
        # print(f"[Engine] model.transformer_layers[0]: {self.model.transformer_layers[0]}")
        # print(f"[Engine] model.transformer_layers[0] vars: {vars(self.model.transformer_layers[0])}")
        # update_kvcache_after_replacing_layer(self, layer_id=layer_id)

        # print(f"\n[Engine] Restore the original layer {layer_id}...")
        # if if_kv_cache_unused(self, layer_id=layer_id):
        #     print(f"[Engine] KV cache with layer_id {layer_id} is unused, can restore the original layer in place")
        #     restore_layer_in_place(self, layer_id=layer_id)
        #     print(f"[Engine] After restoring layer {layer_id}:")
        #     print(f"[Engine] model.transformer_layers length: {len(self.model.transformer_layers)}")
        #     print(f"[Engine] model.transformer_layers[0]: {self.model.transformer_layers[0]}")
        #     print(f"[Engine] model.transformer_layers[0] vars: {vars(self.model.transformer_layers[0])}")
        #     print(f"[Engine] model.transformer_layers[0].weight vars: {vars(self.model.transformer_layers[0].weight)}")
        # else:
        #     print(f"[Engine] KV cache with layer_id {layer_id} is used, cannot restore the original layer in place")
    

    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request (streaming mode)
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break
    
    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        Add a raw request to the engine and wait for the completion (non-streaming mode)
        Return the output token ids
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def _tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.002)
                continue
            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []
            new_requests = []
            if cur_untokenized_raw_requests[0][0].prompt_token_ids:
                tokenized_time = time.perf_counter()
                for (request, _) in cur_untokenized_raw_requests:
                    request.req_metrics.tokenized_time = tokenized_time
                    new_requests.append(request)
            else:
                prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
                prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)
                tokenized_time = time.perf_counter()  # Record tokenized time   
                for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                    request.prompt_token_ids = prompt_token_id
                    request.prompt_len = len(prompt_token_id)
                    request.req_metrics.tokenized_time = tokenized_time
                    new_requests.append(request)
            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop
    
    async def _main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            try:
                # # Check if the GPU KV cache usage is greater than the threshold and if any layer is not quantized
                # gpu_kv_cache_usage = 1 - (self.model.gpu_block_manager.num_free_blocks / self.model.gpu_block_manager.num_blocks)
                # if gpu_kv_cache_usage > self.engine_config.gpu_kv_cache_threshold:
                #     if len(self.model.layer_quant_list) < self.engine_config.num_layers_to_quantize:
                #         print(f"\n[Engine] GPU KV cache usage: {gpu_kv_cache_usage*100:.2f}%, which is greater than the threshold: {self.engine_config.gpu_kv_cache_threshold*100:.2f}%, And len(self.model.layer_quant_list): {len(self.model.layer_quant_list)}, num_layers_to_quantize: {self.engine_config.num_layers_to_quantize}")
                #         org2quant_and_update_kvcache(self)
                # elif gpu_kv_cache_usage <= 0.5 and len(self.model.layer_quant_list) > 0:
                #     print(f"\n[Engine] GPU KV cache usage: {gpu_kv_cache_usage*100:.2f}%, which is greater than the threshold: {self.engine_config.gpu_kv_cache_threshold*100:.2f}%")
                    
                # Get the next batch from the scheduler
                cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch(engine=self)
                if not cur_batch and not cur_swap_in and not cur_swap_out:
                    # No new batch, sleep for a bit
                    await asyncio.sleep(0.005)
                    continue

                # Perform swap in/out
                # If there are requests in cur_swap_out, the await statement:
                # Calls self._run_on_model_async(...).
                # Suspends the _main_event_loop coroutine until swap_out_seqs finishes.
                # Meanwhile, other background tasks (like metrics recording or tokenization) 
                # can continue to run asynchronously.
                if cur_swap_out:
                    self.engine_config.req_preempt += len(cur_swap_out)
                    org_num_free_blocks = self.model.gpu_block_manager.num_free_blocks.item()
                    time_start = time.perf_counter()
                    await self._run_on_model_async(
                        self.model.swap_out_seqs,
                        [req.request_id for req in cur_swap_out]
                    )
                    print(f"[Engine] Swapping out {len(cur_swap_out)} requests with time: {(time.perf_counter() - time_start)*1000:.2f} ms, num_free_blocks: {org_num_free_blocks} -> {self.model.gpu_block_manager.num_free_blocks}.")
                if cur_swap_in:
                    org_num_free_blocks = self.model.gpu_block_manager.num_free_blocks.item()
                    time_start = time.perf_counter()
                    await self._run_on_model_async(
                        self.model.swap_in_seqs,
                        [req.request_id for req in cur_swap_in]
                    )
                    print(f"[Engine] Swapping in {len(cur_swap_in)} requests with time: {(time.perf_counter() - time_start)*1000:.2f} ms, num_free_blocks: {org_num_free_blocks} -> {self.model.gpu_block_manager.num_free_blocks}.")
                # Forward the model
                # there is no output tokens yet, the the req is still in prefill stage, 
                # so that the input_ids would be the whole prompt_token_ids;
                # Or if there are some output tokens, so the input id is just only the last generated token.
                # input_ids shape (len(seq), len(input_ids_of_each_seq))
                input_ids = [
                    req.prompt_token_ids if req.is_prefill_stage() else [req.output_token_ids[-1]]
                    for req in cur_batch
                ]
                seq_ids = [req.request_id for req in cur_batch]
                decoding_seq_lens_list = [
                    req.prompt_len + req.get_cur_output_len()
                    for req in cur_batch
                    if not req.is_prefill_stage()
                ]

                # if len(input_ids[0]) == 11:
                #     print("~"*50, '[Engine] Replacing transformer layers', "~"*50)
                #     print("Before replacing transformer layer")
                #     print(f"[Engine] Org Model: {self.model}")
                #     print(f"[Engine] Org Model transformer_layers length: {len(self.model.transformer_layers)}")
                #     print(f"[Engine] Org Model transformer_layers: {self.model.transformer_layers}")
                #     print(f"[Engine] Org Model transformer_layers_0: {self.model.transformer_layers[0]}")
                #     print(f"[Engine] Org Model transformer_layers_0 vars: {vars(self.model.transformer_layers[0])}")
                #     print(f"[Engine] Org Model transformer_layers_0 weight: {self.model.transformer_layers[0].weight}")
                #     print(f"[Engine] Org Model transformer_layers_0 weight vars: {vars(self.model.transformer_layers[0].weight)}")
        
                #     # replace_layer_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                #     # replace_layer_ids = list(range(0, 32))
                #     # replace_layer_ids = [0]
                #     # for layer_id in replace_layer_ids:

                    # swap_layer_in_place(self, layer_id=0)
                #     print("\nAfter replacing transformer layer")
                #     print(f"[Engine] Org Model: {self.model}")
                #     print(f"[Engine] Org Model transformer_layers length: {len(self.model.transformer_layers)}")
                #     print(f"[Engine] Org Model transformer_layers: {self.model.transformer_layers}")
                #     print(f"[Engine] Org Model transformer_layers_0: {self.model.transformer_layers[0]}")
                #     print(f"[Engine] Org Model transformer_layers_0 vars: {vars(self.model.transformer_layers[0])}")
                #     print(f"[Engine] Org Model transformer_layers_0_awq_layer: {self.model.transformer_layers[0].awq_layer}")
                #     print(f"[Engine] Org Model transformer_layers_0_awq_layer vars: {vars(self.model.transformer_layers[0].awq_layer)}")
                #     layers_in_dict = self.model.transformer_layers[0].awq_layer.state_dict()
                #     for key, value in layers_in_dict.items():
                #         print(f"[Engine] {key}: {value}")

                    # update_kvcache_after_replacing_layer(self, layer_id=0)
                #     print("\nAfter updating kvcache")
                #     print(f"[Engine] Org Model: {self.model}")
                #     print(f"[Engine] Org Model transformer_layers length: {len(self.model.transformer_layers)}")
                #     print(f"[Engine] Org Model transformer_layers: {self.model.transformer_layers}")
                #     print(f"[Engine] Org Model transformer_layers_0: {self.model.transformer_layers[0]}")
                #     print(f"[Engine] Org Model transformer_layers_0 vars: {vars(self.model.transformer_layers[0])}")
                #     print(f"[Engine] Org Model transformer_layers_0_awq_layer: {self.model.transformer_layers[0].awq_layer}")
                #     print(f"[Engine] Org Model transformer_layers_0_awq_layer vars: {vars(self.model.transformer_layers[0].awq_layer)}")
                #     layers_in_dict = self.model.transformer_layers[0].awq_layer.state_dict()
                #     for key, value in layers_in_dict.items():
                #         print(f"[Engine] {key}: {value}")

                # Record timing for throughput calculation
                start_time = time.perf_counter()
                output_tokens = await self._run_on_model_async(
                    self.model.forward,
                    input_ids,
                    seq_ids,
                    decoding_seq_lens_list
                )
                end_time = time.perf_counter()

                # if self.num_layers_quantized > 0:
                #     print(f"[Engine] len(self.scheduler.running_q): {len(self.scheduler.running_q)}, num_layers_quantized: {self.num_layers_quantized}")
                #     for req in cur_batch:
                #         print(f"[Engine] req.request_id: {req.request_id}, req.is_prefill_stage(): {req.is_prefill_stage()}, req.prompt_len: {req.prompt_len}, req.output_len: {req.output_len}, req.get_cur_output_len(): {req.get_cur_output_len()}, len(req.output_token_ids): {len(req.output_token_ids)}")
                #     exit()
                
                # print("~"*100)
                # print("len(output_tokens):", len(output_tokens))
                # print("output_tokens[0].shape:", output_tokens[0].shape)
                # print("output_tokens[0]:", output_tokens[0])
                # print("~"*100)
                # Update throughput metrics
                for req in cur_batch:
                    if req.is_prefill_stage():
                        self.server_metrics.prefill_tokens += req.prompt_len
                        self.server_metrics.prefill_time += end_time - start_time
                    else:
                        self.server_metrics.decode_tokens += 1
                        self.server_metrics.decode_time += end_time - start_time

                # Deal with output tokens
                finished_req_ids = []
                completion_time = time.perf_counter()
                for req, output_token in zip(cur_batch, output_tokens):
                    if req.is_prefill_stage():
                        req.req_metrics.first_token_time = completion_time
                    req.output_token_ids.append(output_token)
                    req.num_layers_quantized = max(len(self.model.layer_quant_list), req.num_layers_quantized)
                    req.output_q.put_nowait(StepOutput(output_token, req))
                    if req.is_finished():
                        finished_req_ids.append(req.request_id)
                        req.req_metrics.completion_time = completion_time
                        req.finished_event.set()
                
                await self._run_on_model_async(
                    self.model.free_seqs_resources,
                    finished_req_ids
                )
                
                # Inform the scheduler
                self.scheduler.on_batch_finish(cur_batch)

            except Exception as e:
                print(f"Error in main event loop: {e}")
                # await asyncio.sleep(1)

    
    async def _metrics_event_loop(self):
        """
        Dedicated event loop for collecting and recording metrics.
        Runs at regular intervals defined by self.metrics_interval.
        """
        start_time = time.perf_counter()
        while True:
            try:
                # Get current metrics
                num_requests_waiting = len(self.scheduler.waiting_q)
                num_requests_running = len(self.scheduler.running_q)
                num_requests_swapping = len(self.scheduler.swapped_q)
                
                # Get GPU metrics
                gpu_total_blocks = self.model.gpu_block_manager.num_blocks
                gpu_free_blocks = self.model.gpu_block_manager.num_free_blocks
                gpu_kv_cache_usage = float((gpu_total_blocks-gpu_free_blocks) / gpu_total_blocks)
                
                # Get CPU metrics
                total_blocks = self.model.cpu_block_manager.num_blocks
                free_blocks = self.model.cpu_block_manager.num_free_blocks
                cpu_kv_cache_usage = float((total_blocks-free_blocks) / total_blocks)
                
                # Calculate throughput metrics
                if self.server_metrics.prefill_time == 0:
                    prefilling_throughput = 0.0
                else:
                    prefilling_throughput = self.server_metrics.prefill_tokens / self.server_metrics.prefill_time
                if self.server_metrics.decode_time == 0:
                    decoding_throughput = 0.0
                else:
                    decoding_throughput = self.server_metrics.decode_tokens / self.server_metrics.decode_time
                tokens_per_second = prefilling_throughput + decoding_throughput
                
                # Record metrics
                self.server_metrics.record_metrics(
                    num_requests_waiting=num_requests_waiting,
                    num_requests_running=num_requests_running,
                    num_requests_swapping=num_requests_swapping,
                    num_preemptions=self.server_metrics.preemption_count,
                    num_layer_quantized=len(self.model.layer_quant_list),
                    gpu_total_blocks=gpu_total_blocks,
                    gpu_kv_cache_usage=gpu_kv_cache_usage,
                    cpu_kv_cache_usage=cpu_kv_cache_usage,
                    prefilling_throughput=prefilling_throughput,
                    decoding_throughput=decoding_throughput,
                    tokens_per_second=tokens_per_second
                )
                
                # Reset metrics tracking
                self.server_metrics.prefill_tokens = 0
                self.server_metrics.prefill_time = 0
                self.server_metrics.decode_tokens = 0
                self.server_metrics.decode_time = 0
                
                # Log metrics summary periodically if needed
                print(f"[{time.perf_counter() - start_time:.2f}] GPU KV cache usage: {gpu_kv_cache_usage*100:.2f}% ({self.model.gpu_block_manager.num_blocks - self.model.gpu_block_manager.num_free_blocks}/{self.model.gpu_block_manager.num_blocks}); "
                    f"Reqs: {num_requests_running} running, {num_requests_waiting} waiting; "
                    f"Pref thrput: {prefilling_throughput:.2f} tokens/s, "
                    f"Decd thrput: {decoding_throughput:.2f} tokens/s, "
                    f"Total thrput: {tokens_per_second:.2f} tokens/s; "
                    f"Num layers quant: {len(self.model.layer_quant_list)}."
                    f"model.layer_quant_list: {self.model.layer_quant_list}.")

            except Exception as e:
                print(f"Error in metrics event loop: {e}")
                
            # Sleep until next metrics collection interval
            await asyncio.sleep(self.server_metrics.time_interval)


    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."

        # asyncio.gather() runs multiple asynchronous tasks concurrently and waits for all of them to complete.
        # Both _tokenize_raw_request_event_loop() and _main_event_loop() are started in parallel.
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop(),
            self._metrics_event_loop(),
        )