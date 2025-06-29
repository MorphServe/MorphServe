import itertools
import math
import time
import torch
import torch.nn as nn

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import GB, MB
import swiftllm_c

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

class LlamaModel(nn.Module):
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_path: str = None,
    ):
        """
        Initialize the LlamaModel.
        Args:
            engine_config: Engine configuration
            model_path: Path to model weights.
        """
        super().__init__()
        self.engine_config = engine_config

        # Use provided model path or default from engine config
        self.model_path = model_path

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(self.model_path)

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None

        self.block_swap = 0
        
        # KV cache for after quantized layers replacement
        self.k_cache_new = list()
        self.v_cache_new = list()
        self.kv_cache_new_block_size = 0
        self.org_layer_param_size = 0
        self.layer_quant_list = list()
        self.is_layer_quant_list = [False] * self.model_config.num_layers
        self.next_unquantized_layer = self.model_config.num_layers - 1

        
    @torch.inference_mode()
    def load_weights(self, load_to_cpu: bool = False):
        """
        Load weights and initialize layers
        Args:
            load_to_cpu: If True, load weights to CPU memory instead of GPU
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.model_path,
            self.engine_config.use_dummy,
            load_to_cpu=load_to_cpu
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        
        # Load all layers
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.

        The profile_num_blocks() function efficiently profiles GPU memory usage
        by simulating a realistic batch scenario 
        (which is only determinded by the self.engine_config.max_tokens_in_batch)
        and dynamically determining the maximum number of KV cache blocks.

        The function first clear all the unused GPU memory, and perform an prefill simulation
        based on the max_tokens_in_batch, and record the memory peak usage; 
        so that the unused memory during this process can be allocated to KV cache;
        then calculate the number of GPU blocks;
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        print(f"[Model.profile] Initial GPU memory: {initial_memory / (1024**2):.2f} MB")

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, free_memory: {free_memory/GB:.2f} GB, peak_memory: {peak_memory/GB:.2f} GB, useable_memory: {useable_memory/GB:.2f} GB, kv_real_available_memory: {(useable_memory-peak_memory)/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        print(f"[Model.profile] KV_block_size_bytes: {block_size_bytes/MB:.2f} MB, with block_size: {self.engine_config.block_size} and kvslot_size: {self.model_config.get_kvslot_size()/MB:.2f} MB")
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks
    
    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        print(f"[Model.init_kvcache_and_swap] self.k_cache.shape: {self.k_cache.shape}")
        print(f"[Model.init_kvcache_and_swap] self.v_cache.shape: {self.v_cache.shape}")

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

        swiftllm_c.register_kv_cache_info(
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )


    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        # print("~"*18, "LlamaModel._forward() input_ids: shape", input_ids.shape, '~'*18)
        # print(input_ids) 
        # print("~"*18, "LlamaModel._forward() infer_state:", '~'*18)
        # for k, v in vars(infer_state).items():
        #     print(f"{k}: {v}")
        # position_embeddings = (infer_state.position_cos, infer_state.position_sin)
        # print("position_embeddings", position_embeddings)
        # print("position_embeddings[0]", position_embeddings[0], "position_embeddings[0].shape", position_embeddings[0].shape, "position_embeddings[0].numel()", position_embeddings[0].numel())
        # print("position_embeddings[1]", position_embeddings[1], "position_embeddings[1].shape", position_embeddings[1].shape, "position_embeddings[1].numel()", position_embeddings[1].numel())
        
        # 1. Preprocessing: Convert input token IDs to embeddings
        input_embds = self.pre_layer.forward(input_ids)

        # print('\n'+"~"*18, "LlamaModel._forward() input_embds: shape", input_embds.shape, '~'*18)
        # print(input_embds)
        # input_embds are the same as the AWQ (default huggingface.transformers.llama.modeling_llama.LlamaModel.forward, inputs_embeds)

        # 2. Prepare a residual buffer: same shape as input_embds, initialized to zeros.
        residual_buf = torch.zeros_like(input_embds)

        # print('\n' + "~"*18, "LlamaModel._forward() self.transformer_layers:", '~'*18)
        # 3. Process through each transformer layer:
        for layer in self.transformer_layers:
            # Layer is a Hugging Face LlamaDecoderLayer, use the original forward method: Otherwise, use the custom forward method
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                self.k_cache_new,
                self.v_cache_new,
                self.gpu_block_manager.block_table if not infer_state.ignore_kvcache else None,
                infer_state,
            )
            # if cnt <= 1 or cnt >= len(self.transformer_layers) - 2:
            #     print('\n' + str(cnt), "~"*18, "after Decoding.forward, the output of the layer:", '~'*18)
            #     print("input_embds", input_embds, "input_embds.shape", input_embds.shape, "input_embds.numel()", input_embds.numel())
            #     print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
            # cnt += 1
        # print('\n' + "~"*18, "LlamaModel._forward() after all layers, add the residual connection.", '~'*18, '\n')
        # 4. After all layers, add the residual connection. 
        input_embds += residual_buf
        # print('\n' + "~"*18, "LlamaModel._forward() after add the residual connection, call the post layer.", '~'*18, '\n')

        # 5. Postprocessing: Convert final embeddings into output tokens/logits.
        output_tokens = self.post_layer.forward(input_embds, infer_state)
        # print('\n' + "~"*18, "LlamaModel._forward() after call the post layer, return the output tokens.", '~'*18, '\n')
        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]], # [batch_size, *]
        seq_ids_list: list[int],     # [batch_size]
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks
    ) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """

        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list

        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        # Assign None to num_blocks_org if the gpu_block_manager is not initialized during the profile_num_blocks()
        num_blocks_org = self.gpu_block_manager.num_blocks_org if self.gpu_block_manager else None

        # Assign the kv_cache_new_block_size to num_blocks_org
        kv_cache_new_block_size = self.kv_cache_new_block_size
        org_layer_param_size = self.org_layer_param_size

        # Building a 1D tensor of position indices by:
        # 1. Generating all the "running" indices for the prefill sequences (for each prefill sequence, from 0 to length-1).
        # 2. Appending the last index (length minus 1) for each decoding sequence.
        # for example the position_indices: tensor([0, 1, 2, 0, 1, 2, 3, 4, 9, 7])
        # Question: Why the last index of the decoding sequence is length minus 1? why not the length?
        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        # Function allocate_blocks_for_seqs() will allocate blocks for sequences, 
        # and will only ever update the block tables, not the actual KV tensors, 
        # so it is safe to call it even if we have quantized layers
        if not ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                seq_ids,
                seq_lengths
            )

        # Select the seq_block_size
        # Here we use a simple heuristic:
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        # The trade-off is between the number of blocks and the number of threads in the Triton kernel.
        # 1) seq_block_size too large <-> num_seq_blocks too small
        #    - Fewer blocks -> fewer kernel launches -> fewer parallel threads
        #    - But each kernel launch has more work to do:
        #      - More memory reads/writes for each thread
        #      - Less utilization of GPU SMs
        #      + better memory coalescing
        # 2) seq_block_size too small <-> num_seq_blocks too large
        #    - More blocks -> more kernel launches -> more parallel threads
        #    - But each kernel launch has less work to do:
        #      - Might overwhelm shared memory and registers with small tiles
        #      - Less memory coalescing
        #      + Less memory reads/writes for each thread

        seq_block_size = 512
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and \
                                                                seq_block_size//2 >= 64 and \
                                            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2

        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            num_blocks_org = num_blocks_org,
            kv_cache_new_block_size = kv_cache_new_block_size,
            org_layer_param_size = org_layer_param_size,
            
            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache
        )

        # print the debug information during the real inference
        # if not ignore_kvcache:
        #     print('\n' + "~"*18, "LlamaModel.forward", '~'*18)
        #     print("input_ids_list:", input_ids_list)
        #     print("seq_ids_list:", seq_ids_list)
        #     print("decoding_seq_lens_list:", decoding_seq_lens_list)
        #     print("num_prefill_seqs:", num_prefill_seqs)
        #     print("flattened_input_ids:", flattened_input_ids)
        #     print("seq_lengths_list:", seq_lengths_list)
        #     print("seq_ids:", seq_ids)
        #     print("seq_lengths:", seq_lengths)
        #     print("batch_size:", batch_size)
        #     print("num_tokens:", num_tokens)
        #     print("prefill_seq_lens_list:", prefill_seq_lens_list)
        #     print("prefill_seq_lens:", prefill_seq_lens)
        #     print("prefill_start_locs:", prefill_start_locs)
        #     print("max_prefill_len:", max_prefill_len)
        #     print("decoding_seq_lens:", decoding_seq_lens)
        #     print("max_decoding_len:", max_decoding_len)
        #     print("position_indices:", position_indices)
        #     print("ignore_kvcache:", ignore_kvcache)
        #     print('\n' + "~"*18, "infer_state", '~'*18)
        #     for k, v in vars(infer_state).items():
        #         print(f"{k}: {v}")

        return self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state
        ).tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.engine_config.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        self.block_swap += len(src_block_ids)
        # If the max block id of src and dst are both less than the num_blocks_org, use the old version of swap_blocks
        if torch.max(src_block_ids) < self.gpu_block_manager.num_blocks_org and torch.max(dst_block_ids) < self.gpu_block_manager.num_blocks_org:
            swiftllm_c.swap_blocks(
                src_block_ids.tolist(),
                dst_block_ids.tolist(),
                is_swap_in,
                self.k_cache, self.v_cache,
                self.k_swap, self.v_swap
            )
        else:
            # If the max block id of src and dst are both greater than the num_blocks_org, use the new version of swap_blocks
            # print("[Swapping] swap_blocks_new_kv_combined")
            swiftllm_c.swap_blocks_new_kv_combined(
                src_block_ids.tolist(),
                dst_block_ids.tolist(),
                is_swap_in,
                self.gpu_block_manager.num_blocks_org,
                self.org_layer_param_size,
                self.kv_cache_new_block_size,
                self.k_cache.element_size(),

                self.k_cache, self.v_cache,
                self.k_swap, self.v_swap,
                self.k_cache_new[0], self.v_cache_new[0]
            )
            # exit()
    
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
