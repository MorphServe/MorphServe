import torch
import time
import vllm_flash_attn

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache, store_kvcache_performance_test
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace

debug_print = False
debug_cache_attention = False

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        decoding_piggyback_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.layer_id = layer_id
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_cache_new: torch.Tensor,
        v_cache_new: torch.Tensor,
        block_table: torch.Tensor,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:

        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        # print('\n' + "~"*18, "before RMSNorm, LlamaTransformerLayer.forward, input_embds", '~'*18)
        # print("input_embds", input_embds, "input_embds.shape", input_embds.shape, "input_embds.numel()", input_embds.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
        # print("LlamaTransformerLayer.forward, self.model_config.rms_norm_eps", self.model_config.rms_norm_eps)
        # print("LlamaTransformerLayer.forward, self.weight.attn_norm", self.weight.attn_norm, "self.weight.attn_norm.shape", self.weight.attn_norm.shape, "self.weight.attn_norm.numel()", self.weight.attn_norm.numel())
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )
        # print("~"*18, "after RMSNorm, LlamaTransformerLayer.forward, input_embds", '~'*18)
        # print("input_embds", input_embds, "input_embds.shape", input_embds.shape, "input_embds.numel()", input_embds.numel())   
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())

        # print("~"*18, "before QKV, LlamaTransformerLayer.forward weights:", '~'*18)
        # print("q_proj", self.weight.q_proj, "q_proj.shape", self.weight.q_proj.shape, "q_proj.numel()", self.weight.q_proj.numel())
        # print("k_proj", self.weight.k_proj, "k_proj.shape", self.weight.k_proj.shape, "k_proj.numel()", self.weight.k_proj.numel())
        # print("v_proj", self.weight.v_proj, "v_proj.shape", self.weight.v_proj.shape, "v_proj.numel()", self.weight.v_proj.numel())
        
        # Calculate QKV
        q = linear(input_embds, self.weight.q_proj)		# [num_total_tokens, hidden_size]
        k = linear(input_embds, self.weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = linear(input_embds, self.weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        
        # print("~"*18, "after QKV, LlamaTransformerLayer.forward q k v:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        # print("v", v, "v.shape", v.shape, "v.numel()", v.numel())
        
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # print("~"*18, "after QKV view, LlamaTransformerLayer.forward q k v:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        # print("v", v, "v.shape", v.shape, "v.numel()", v.numel())
        
        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        # print("~"*18, "after rotary embedding, LlamaTransformerLayer.forward q k:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        if debug_print and self.layer_id == 0:
            print('\n' + "~"*18, "Layer", self.layer_id, "before store kvcache, LlamaTransformerLayer.forward:", '~'*18)
        # if k_cache is not None:
        #     print("k_cache type", type(k_cache), "k_cache shape", k_cache.shape, "k_cache numel", k_cache.numel())
        # if v_cache is not None:
        #     print("v_cache type", type(v_cache), "v_cache shape", v_cache.shape, "v_cache numel", v_cache.numel())
        # if block_table is not None:
        #     print("block_table type", type(block_table), "block_table shape", block_table.shape, "block_table numel", block_table.numel())

        start_time_store_kvcache = time.perf_counter()

        # Store_kvcache saves the new k, v values into k_cache, v_cache, k_cache_new, v_cache_new.
        # Executed on the default CUDA stream
        # This includes Triton kernels, which are asynchronous on the GPU.
        if not infer_state.ignore_kvcache:
            store_kvcache(
                k, v,   
                k_cache, v_cache,
                k_cache_new, v_cache_new,
                block_table,
                self.model_config,
                self.engine_config,
                infer_state,
                self.layer_id
            )
        # Create an event to synchronize the store_kvcache operation
        store_kvcache_event = torch.cuda.Event()
        # The event is added to the default stream.
        # It will not be considered "done" until all previous ops in that stream complete.
        store_kvcache_event.record()    # <-- Records this point in the default stream

        end_time_store_kvcache = time.perf_counter()
        if debug_print and self.layer_id == 0:
            print("~"*18, "Layer", self.layer_id, "after store kvcache, LlamaTransformerLayer.forward continue...", '~'*18)

        # print("~"*18, "after store kvcache, LlamaTransformerLayer.forward:", '~'*18)
        # if k_cache is not None:
        #     print("k_cache type", type(k_cache), "k_cache shape", k_cache.shape, "k_cache numel", k_cache.numel())
        # if v_cache is not None:
        #     print("v_cache type", type(v_cache), "v_cache shape", v_cache.shape, "v_cache numel", v_cache.numel())
        # if block_table is not None:
        #     print("block_table type", type(block_table), "block_table shape", block_table.shape, "block_table numel", block_table.numel())
        
        # Attention
        o = input_embds    # [num_total_tokens, hidden_size]
        # print("~"*18, "before attention, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        # During prefill stage, there is no use of KV cache,
        # so we can directly use the scripts for both original and new kv caches.
        if infer_state.num_prefill_seqs > 0:
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
                q[:infer_state.num_prefill_tokens, :, :],
                k[:infer_state.num_prefill_tokens, :, :],
                v[:infer_state.num_prefill_tokens, :, :],
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.max_prefill_len,
                infer_state.max_prefill_len,
                softmax_scale=infer_state.softmax_scale,
                causal=True
            ).reshape(-1, self.model_config.hidden_size)

        end_time_prefill_attention = time.perf_counter()
        if debug_print and self.layer_id == 0:
            print("~"*18, "Layer", self.layer_id, "after prefill attention, LlamaTransformerLayer.forward continue...", '~'*18)
        
        # TODO: during decoding stage, there are KV caches used,
        #       we need to handle the original and new kv caches separately.
        end_time_wait_store_kvcache_event = 0
        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            # Create a new CUDA stream for decoding piggyback operations, which is not the default stream!
            # This stream is used for operations that need to wait for the store_kvcache operation to complete
            with torch.cuda.stream(self.decoding_piggyback_stream):
                # Wait for the store_kvcache operation to complete (up to the store_kvcache_event.record() call) on the default stream
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                end_time_wait_store_kvcache_event = time.perf_counter()
                paged_attention(
                    q[infer_state.num_prefill_tokens:, :, :],
                    k_cache, v_cache, k_cache_new, v_cache_new, block_table,
                    self.model_config, self.engine_config, infer_state,
                    self.layer_id,
                    o[infer_state.num_prefill_tokens:, :],
                )
                event = torch.cuda.Event()
                event.record()
            # Wait for the decoding piggyback operation to complete on the default stream
            torch.cuda.default_stream().wait_event(event)
        
        end_time_decoding_attention = time.perf_counter()
        if debug_print and self.layer_id == 0:
            print("~"*18, "Layer", self.layer_id, "after decoding attention, LlamaTransformerLayer.forward continue...", '~'*18)
        
        time_threshold = 0.5 # 500 ms
        if end_time_decoding_attention - start_time_store_kvcache > time_threshold:
            print(f"\n[LlamaTransformerLayer-Serving] Layer {self.layer_id} serving time exceeded {time_threshold} s with {(end_time_decoding_attention - start_time_store_kvcache):.2f} s.")
            print(f"[LlamaTransformerLayer-Serving] Time to store kvcache:       {(end_time_store_kvcache - start_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to prefill attention:   {(end_time_prefill_attention - end_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to store kvcache event: {(end_time_wait_store_kvcache_event - start_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to decode attention:    {(end_time_decoding_attention - end_time_wait_store_kvcache_event)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Total time:                  {(end_time_decoding_attention - start_time_store_kvcache)*1000:.2f} ms.\n")

        # Output GEMM
        o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]
        # print("~"*18, "after attention and GEMM, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())

        # print("~"*18, "before FFN norm, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
        # print("self.weight.ffn_norm", self.weight.ffn_norm, "self.weight.ffn_norm.shape", self.weight.ffn_norm.shape, "self.weight.ffn_norm.numel()", self.weight.ffn_norm.numel())
        
        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        q = None
        k = None
        v = None

        # print("~"*18, "after FFN norm, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())

        # FFN
        # print("~"*18, "before FFN, LlamaTransformerLayer.forward:", '~'*18)
        # print("weight: up_gate_proj", self.weight.up_gate_proj, "up_gate_proj.shape", self.weight.up_gate_proj.shape, "up_gate_proj.numel()", self.weight.up_gate_proj.numel())
        # print("weight: down_proj", self.weight.down_proj, "down_proj.shape", self.weight.down_proj.shape, "down_proj.numel()", self.weight.down_proj.numel())

        up_gate_proj = linear(o, self.weight.up_gate_proj)
        # print("~"*18, "intermediate after o * up_gate_proj:", '~'*18)
        # print("up_gate_proj", up_gate_proj, "up_gate_proj.shape", up_gate_proj.shape, "up_gate_proj.numel()", up_gate_proj.numel())
        silu_and_mul_inplace(up_gate_proj)
        # print("~"*18, "intermediate after silu_and_mul_inplace:", '~'*18)
        # print("up_gate_proj", up_gate_proj, "up_gate_proj.shape", up_gate_proj.shape, "up_gate_proj.numel()", up_gate_proj.numel())
        ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        # print("~"*18, "final ffn_out:", '~'*18)
        # print("ffn_out", ffn_out, "ffn_out.shape", ffn_out.shape, "ffn_out.numel()", ffn_out.numel())

        return ffn_out
    

class AWQTransformerLayer():
    def __init__(self, 
                 awq_layer, 
                 model_config: LlamaModelConfig, 
                 engine_config: EngineConfig, 
                 layer_id: int, 
                 decoding_piggyback_stream: torch.cuda.Stream): 
        self.awq_layer = awq_layer
        self.model_config = model_config
        self.engine_config = engine_config
        self.layer_id = layer_id
        self.decoding_piggyback_stream = decoding_piggyback_stream

    def forward(
            self, 
            input_embds: torch.Tensor,
            residual_buf: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            k_cache_new: torch.Tensor,
            v_cache_new: torch.Tensor,
            block_table: torch.Tensor,
            infer_state: LlamaInferState
        ) -> torch.Tensor:

        # print("~"*18, "before RMSNorm, LlamaTransformerLayer.forward, input_embds", '~'*18)
        # print("input_embds", input_embds, "input_embds.shape", input_embds.shape, "input_embds.numel()", input_embds.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
        # print("self.awq_layer", self.awq_layer)
        # print("self.awq_layer.input_layernorm vars", vars(self.awq_layer.input_layernorm))
        # print("self.awq_layer.input_layernorm.weight", self.awq_layer.input_layernorm.weight, "self.awq_layer.input_layernorm.weight.shape", self.awq_layer.input_layernorm.weight.shape, "self.awq_layer.input_layernorm.weight.numel()", self.awq_layer.input_layernorm.weight.numel())
        # print("self.model_config.rms_norm_eps", self.model_config.rms_norm_eps)
        
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.awq_layer.input_layernorm.weight,
            self.model_config.rms_norm_eps
        )
        # print("~"*18, "after RMSNorm, LlamaTransformerLayer.forward, input_embds", '~'*18)
        # print("input_embds", input_embds, "input_embds.shape", input_embds.shape, "input_embds.numel()", input_embds.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
        # print("~"*50)
        
        # Calculate QKV using AWQ quantized layers
        # Create input shape expected by AWQ layers - reshaping from [num_tokens, hidden_size] 
        # to [batch_size, seq_len, hidden_size]
        # For inference, we can reshape as [1, num_tokens, hidden_size]
        batch_size = 1
        seq_len = input_embds.size(0)
        hidden_size = input_embds.size(1)
        # Reshape for AWQ layer inputs
        reshaped_input = input_embds.view(batch_size, seq_len, hidden_size)
        
        # print("~"*18, "before QKV, LlamaTransformerLayer.forward weights:", '~'*18)
        # print("q_proj", self.awq_layer.self_attn.q_proj, "\n", "vars(self.awq_layer.self_attn.q_proj)", vars(self.awq_layer.self_attn.q_proj))
        # print("k_proj", self.awq_layer.self_attn.k_proj, "\n", "vars(self.awq_layer.self_attn.k_proj)", vars(self.awq_layer.self_attn.k_proj))
        # print("v_proj", self.awq_layer.self_attn.v_proj, "\n", "vars(self.awq_layer.self_attn.v_proj)", vars(self.awq_layer.self_attn.v_proj))

        # Use AWQ quantized projections
        q = self.awq_layer.self_attn.q_proj(reshaped_input)  # [1, seq_len, hidden_size]
        k = self.awq_layer.self_attn.k_proj(reshaped_input)  # [1, seq_len, kv_dim]
        v = self.awq_layer.self_attn.v_proj(reshaped_input)  # [1, seq_len, kv_dim]
        
        # print("~"*18, "after QKV, LlamaTransformerLayer.forward q k v:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        # print("v", v, "v.shape", v.shape, "v.numel()", v.numel())

        # Reshape back to SwiftLLM expected format and split into heads
        q = q.view(seq_len, self.model_config.num_q_heads, self.model_config.head_dim)
        k = k.view(seq_len, self.model_config.num_kv_heads, self.model_config.head_dim)
        v = v.view(seq_len, self.model_config.num_kv_heads, self.model_config.head_dim)

        # print("~"*18, "after QKV view, LlamaTransformerLayer.forward q k v:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        # print("v", v, "v.shape", v.shape, "v.numel()", v.numel())
        
        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        # print("~"*18, "after rotary embedding, LlamaTransformerLayer.forward q k:", '~'*18)
        # print("q", q, "q.shape", q.shape, "q.numel()", q.numel())
        # print("k", k, "k.shape", k.shape, "k.numel()", k.numel())
        
        if debug_print and self.layer_id == 0:
            print('\n'+"~"*18, "AWQLayer", self.layer_id, "before store kvcache, LlamaTransformerLayer.forward:", '~'*18)
        
        # Debugging for KV Cache Store with Original and New KV Cache
        # key: sequence_id at original block_table, from 0-127
        # value: dict
        #   key_value_1 "seq_id_index": index at the infer_state.seq_ids
        #   key_value_2 "first_org_block_index": index at the original block_table
        #   key_value_3 "first_new_block_index": index exceed the original kv cache size, should be at the new block_table
        kv_cache_debug_dict = {}
        if debug_cache_attention:
            if len(k_cache_new) > 0:
                # Debugging for decoding attention
                if infer_state.num_decoding_seqs > 0:
                    num_blocks_org = infer_state.num_blocks_org
                    mask0 = (block_table >= num_blocks_org)
                    if mask0.any(): 
                        print(f"\n~~~~ store_kvcache layer {self.layer_id}, some blocks exceed the original kv cache size ~~~~~")
                        print(f"k shape: {k.shape}")
                        print(f"k_cache shape: {k_cache.shape}")
                        print(f"k_cache_new[0] shape: {k_cache_new[0].shape}")
                        print(f"~~~~ infer_state ~~~~~")
                        for key, value in infer_state.__dict__.items():
                            if key == "position_cos" or key == "position_sin":
                                print(f"{key}: {value.shape}")
                            else:
                                print(f"{key}: {value}")
                        for index in range(block_table.shape[0]):
                            if num_blocks_org <= max(block_table[index]):
                                print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}] first 256: {block_table[index][:256]}")
                                seq_id_index = (infer_state.seq_ids == index).nonzero(as_tuple=True)[0].item()
                                for block_index in block_table[index]:
                                    if block_index < num_blocks_org:
                                        first_org_block_index = block_index
                                        break
                                for block_index in block_table[index]:
                                    if block_index >= num_blocks_org:
                                        first_new_block_index = block_index
                                        break
                                for block_index in reversed(block_table[index]):
                                    if block_index >= num_blocks_org:
                                        last_new_block_index = block_index
                                        break
                                kv_cache_debug_dict[index] = {
                                    "seq_id_index": seq_id_index,
                                    "first_org_block_index": first_org_block_index,
                                    "first_new_block_index": first_new_block_index,
                                    "last_new_block_index": last_new_block_index
                                }
        # if len(kv_cache_debug_dict) > 0:
        #     print(f"\n~~~~ Before KV cache update, kv_cache_debug_dict ~~~~~")
        #     for seq_id, kv_cache_dict in kv_cache_debug_dict.items():
        #         seq_id_index = kv_cache_dict['seq_id_index']
        #         first_org_block_index = kv_cache_dict['first_org_block_index']
        #         first_new_block_index = kv_cache_dict['first_new_block_index']
        #         last_new_block_index = kv_cache_dict['last_new_block_index']
        #         print(f"~~ seq_id: {seq_id} ~~")
        #         # if now we are in decoding, the seq_id_index can be used to get the K and V (and there is only one K and V);
        #         # and the k[{seq_id_index}][0] would be the first head of the K of this request (later we will print with the kv_cache_new);
        #         print(f"~~ seq_id_index: {seq_id_index} ~~")
        #         print(f"k[{seq_id_index}].shape: {k[seq_id_index].shape}")
        #         print(f"k[{seq_id_index}][0]: {k[seq_id_index][0]}")
        #         print(f"~~ first_org_block_index: {first_org_block_index} ~~")
        #         print(f"k_cache[{first_org_block_index}].shape: {k_cache[first_org_block_index].shape}")
        #         print(f"k_cache[{first_org_block_index}][{self.layer_id}][0][0]: {k_cache[first_org_block_index][self.layer_id][0][0]}")
        #         # if now we are in prefill, the above is only about the first K of the request; (we have printed with the KV_original above)
        #         # and we have many Ks and Vs should be stored in the k_cache and v_cache for this request; (we will print with the KV_new below)
        #         if infer_state.num_prefill_seqs == 1 and infer_state.num_decoding_seqs == 0:
        #             print(f"~"*18, f"prefill with num_tokens: {infer_state.num_prefill_tokens}, no decoding. ~~~~~")
        #             past_blocks = infer_state.num_prefill_tokens // self.engine_config.block_size
        #             past_tokens = past_blocks * self.engine_config.block_size
        #             rest_tokens = infer_state.num_prefill_tokens % self.engine_config.block_size
        #             print(f"KV: past_blocks: {past_blocks}, past_tokens: {past_tokens}, rest_tokens: {rest_tokens}")
        #             kv_cache_new_map_index_last_block = last_new_block_index - num_blocks_org
        #             print(f"KV_cache_new: last_new_block_index: {last_new_block_index}, num_blocks_org: {num_blocks_org}, kv_cache_new_map_index_last_block: {kv_cache_new_map_index_last_block} ~~")
        #             for idx in range(rest_tokens):
        #                 print(f"idx: {idx}, k[{past_tokens + idx}][0]: {k[past_tokens + idx][0]}")
        #                 print(f"idx: {idx}, k_cache_new[0][{kv_cache_new_map_index_last_block}][{self.layer_id}][0][{idx}]: {k_cache_new[0][kv_cache_new_map_index_last_block][self.layer_id][0][idx]}")
        #         else:
        #             # if now we are in decoding, the kv_cache_new is the same as the first K of the request;
        #             kv_cache_new_map_index_first_block = first_new_block_index - num_blocks_org
        #             print(f"\n~~ first_new_block_index: {first_new_block_index}, num_blocks_org: {num_blocks_org}, kv_cache_new_map_index_first_block: {kv_cache_new_map_index_first_block} ~~")
        #             print(f"k_cache_new[0][{kv_cache_new_map_index_first_block}].shape: {k_cache_new[0][kv_cache_new_map_index_first_block].shape}")
        #             print(f"k_cache_new[0][{kv_cache_new_map_index_first_block}][{self.layer_id}][0][0]: {k_cache_new[0][kv_cache_new_map_index_first_block][self.layer_id][0][0]}")

        # if k_cache is not None:
        #     print("k_cache type", type(k_cache), "k_cache shape", k_cache.shape, "k_cache numel", k_cache.numel())
        # if v_cache is not None:
        #     print("v_cache type", type(v_cache), "v_cache shape", v_cache.shape, "v_cache numel", v_cache.numel())
        # if block_table is not None:
        #     print("block_table type", type(block_table), "block_table shape", block_table.shape, "block_table numel", block_table.numel())
        start_time_store_kvcache = time.perf_counter()

        if not infer_state.ignore_kvcache:
            # print("~"*18, "storing kvcache, LlamaTransformerLayer.forward:", '~'*18)
            store_kvcache(
                k, v,
                k_cache, v_cache,
                k_cache_new, v_cache_new,
                block_table,
                self.model_config,
                self.engine_config,
                infer_state,
                self.layer_id
            )
        store_kvcache_event = torch.cuda.Event()
        store_kvcache_event.record()

        end_time_store_kvcache = time.perf_counter()
        if debug_print and self.layer_id == 0:
            print("~"*18, "AWQLayer", self.layer_id, "after store kvcache, LlamaTransformerLayer.forward:", '~'*18)
        # if k_cache is not None:
        #     print("k_cache type", type(k_cache), "k_cache shape", k_cache.shape, "k_cache numel", k_cache.numel())
        # if v_cache is not None:
        #     print("v_cache type", type(v_cache), "v_cache shape", v_cache.shape, "v_cache numel", v_cache.numel())
        # if block_table is not None:
        #     print("block_table type", type(block_table), "block_table shape", block_table.shape, "block_table numel", block_table.numel())
        
        # Attention
        o = input_embds    # [num_total_tokens, hidden_size]

        # Debugging for attention
        if debug_cache_attention:
            if len(kv_cache_debug_dict) > 0 and infer_state.num_decoding_seqs > 0:
                print(f"\n~~~~ Before attention, Debugging for attention ~~~~~")
                print(f"q.shape: {q.shape}")
                print(f"k.shape: {k.shape}")
                print(f"v.shape: {v.shape}")
                print(f"o.shape: {o.shape}")
                decoding_o = o[infer_state.num_prefill_tokens:, :]
                print(f"decoding_o length: {len(decoding_o)}, decoding_o with first 1 request and first 10 elements:")
                print(f"decoding_o[0][:10]: {decoding_o[0][:10]}")
    

        # print("~"*18, "before attention, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        if infer_state.num_prefill_seqs > 0:
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
                q[:infer_state.num_prefill_tokens, :, :],
                k[:infer_state.num_prefill_tokens, :, :],
                v[:infer_state.num_prefill_tokens, :, :],
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.max_prefill_len,
                infer_state.max_prefill_len,
                softmax_scale=infer_state.softmax_scale,
                causal=True
            ).reshape(-1, self.model_config.hidden_size)
            # prefill_attention(
            #     q, k, v, o[:infer_state.num_prefill_tokens, :],
            #     self.model_config, self.engine_config, infer_state
            # )
        end_time_prefill_attention = time.perf_counter()
        if debug_print and self.layer_id == 0:
            print("~"*18, "AWQLayer", self.layer_id, "after prefill attention, LlamaTransformerLayer.forward continue...", '~'*18)
        

        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                end_time_wait_store_kvcache_event = time.perf_counter()
                paged_attention(
                    q[infer_state.num_prefill_tokens:, :, :],
                    k_cache, v_cache, k_cache_new, v_cache_new, block_table,
                    self.model_config, self.engine_config, infer_state,
                    self.layer_id,
                    o[infer_state.num_prefill_tokens:, :],
                )
                event = torch.cuda.Event()
                event.record()
            torch.cuda.default_stream().wait_event(event)
        else:
            # if there is no decoding, we can time here
            end_time_wait_store_kvcache_event = time.perf_counter()
        
        if debug_print and self.layer_id == 0:
            print("~"*18, "AWQLayer", self.layer_id, "after decoding attention, LlamaTransformerLayer.forward continue...", '~'*18)
        # Output GEMM
        # o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]
        end_time_decoding_attention = time.perf_counter()
        time_threshold = 0.5 # 500 ms
        if end_time_decoding_attention - start_time_store_kvcache > time_threshold:
            print(f"\n[LlamaTransformerLayer-Serving] AWQ Layer {self.layer_id} serving time exceeded {time_threshold} s with {(end_time_decoding_attention - start_time_store_kvcache):.2f} s.")
            print(f"[LlamaTransformerLayer-Serving] Time to store kvcache:       {(end_time_store_kvcache - start_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to prefill attention:   {(end_time_prefill_attention - end_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to store kvcache event: {(end_time_wait_store_kvcache_event - start_time_store_kvcache)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Time to decode attention:    {(end_time_decoding_attention - end_time_wait_store_kvcache_event)*1000:.2f} ms.")
            print(f"[LlamaTransformerLayer-Serving] Total time:                  {(end_time_decoding_attention - start_time_store_kvcache)*1000:.2f} ms.")

        # Debugging for Decoding Attention
        if debug_cache_attention:
            if len(kv_cache_debug_dict) > 0:
                # # Debugging for Decoding Attention
                if infer_state.num_decoding_seqs > 0:
                    # There are decoding sequences in this batch
                    print(f"\n~~~~ After Decoding Attention, Debugging for Decoding Attention ~~~~~")
                    decoding_o = o[infer_state.num_prefill_tokens:, :]
                    print(f"decoding_o length: {len(decoding_o)}, decoding_o with first 1 request and first 10 elements:")
                    print(f"decoding_o[0][:10]: {decoding_o[0][:10]}")
                    for seq_id, kv_cache_dict in kv_cache_debug_dict.items():
                        seq_id_index = kv_cache_dict['seq_id_index']
                        print(f"~~ seq_id: {seq_id}, seq_id_index: {seq_id_index} ~~")
                    print("\n\n")
                    exit()
                    

            # # Debugging for KV Cache Store with Original and New KV Cache
            # print(f"\n~~~~ After KV cache update, kv_cache_debug_dict ~~~~~")
            # for seq_id, kv_cache_dict in kv_cache_debug_dict.items():
            #     seq_id_index = kv_cache_dict['seq_id_index']
            #     first_org_block_index = kv_cache_dict['first_org_block_index']
            #     first_new_block_index = kv_cache_dict['first_new_block_index']
            #     last_new_block_index = kv_cache_dict['last_new_block_index']
            #     print(f"~~ seq_id: {seq_id} ~~")
            #     # if now we are in decoding, the seq_id_index can be used to get the K and V (and there is only one K and V);
            #     # and the k[{seq_id_index}][0] would be the first head of the K of this request (later we will print with the kv_cache_new);
            #     print(f"~~ seq_id_index: {seq_id_index} ~~")
            #     print(f"k[{seq_id_index}].shape: {k[seq_id_index].shape}")
            #     print(f"k[{seq_id_index}][0]: {k[seq_id_index][0]}")
            #     print(f"~~ first_org_block_index: {first_org_block_index} ~~")
            #     print(f"k_cache[{first_org_block_index}].shape: {k_cache[first_org_block_index].shape}")
            #     print(f"k_cache[{first_org_block_index}][{self.layer_id}][0][0]: {k_cache[first_org_block_index][self.layer_id][0][0]}")
            #     # if now we are in prefill, the above is only about the first K of the request; (we have printed with the KV_original above)
            #     # and we have many Ks and Vs should be stored in the k_cache and v_cache for this request; (we will print with the KV_new below)
            #     if infer_state.num_prefill_seqs == 1 and infer_state.num_decoding_seqs == 0:
            #         print(f"~"*18, f"prefill with num_tokens: {infer_state.num_prefill_tokens}, no decoding. ~~~~~")
            #         past_blocks = infer_state.num_prefill_tokens // self.engine_config.block_size
            #         past_tokens = past_blocks * self.engine_config.block_size
            #         rest_tokens = infer_state.num_prefill_tokens % self.engine_config.block_size
            #         print(f"KV: past_blocks: {past_blocks}, past_tokens: {past_tokens}, rest_tokens: {rest_tokens}")
            #         kv_cache_new_map_index_last_block = last_new_block_index - num_blocks_org
            #         print(f"KV_cache_new: last_new_block_index: {last_new_block_index}, num_blocks_org: {num_blocks_org}, kv_cache_new_map_index_last_block: {kv_cache_new_map_index_last_block} ~~")
            #         for idx in range(rest_tokens):
            #             print(f"idx: {idx}, k[{past_tokens + idx}][0]: {k[past_tokens + idx][0]}")
            #             print(f"idx: {idx}, k_cache_new[0][{kv_cache_new_map_index_last_block}][{self.layer_id}][0][{idx}]: {k_cache_new[0][kv_cache_new_map_index_last_block][self.layer_id][0][idx]}")
            #     else:
            #         # if now we are in decoding, the kv_cache_new is the same as the first K of the request;
            #         kv_cache_new_map_index_first_block = first_new_block_index - num_blocks_org
            #         print(f"\n~~ first_new_block_index: {first_new_block_index}, num_blocks_org: {num_blocks_org}, kv_cache_new_map_index_first_block: {kv_cache_new_map_index_first_block} ~~")
            #         print(f"k_cache_new[0][{kv_cache_new_map_index_first_block}].shape: {k_cache_new[0][kv_cache_new_map_index_first_block].shape}")
            #         print(f"k_cache_new[0][{kv_cache_new_map_index_first_block}][{self.layer_id}][0][0]: {k_cache_new[0][kv_cache_new_map_index_first_block][self.layer_id][0][0]}")
            #     print("\n\n")
            #     exit()


        # Reshape for AWQ output projection
        reshaped_o = o.view(batch_size, seq_len, hidden_size)
        # Apply AWQ output projection
        o_proj = self.awq_layer.self_attn.o_proj(reshaped_o)
        # Reshape back to SwiftLLM format
        o = o_proj.view(seq_len, hidden_size)

        # print("~"*18, "after attention and GEMM, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())

        # print("~"*18, "before FFN norm, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())
        # print("self.awq_layer.post_attention_layernorm.weight", self.awq_layer.post_attention_layernorm.weight, "self.awq_layer.post_attention_layernorm.weight.shape", self.awq_layer.post_attention_layernorm.weight.shape, "self.awq_layer.post_attention_layernorm.weight.numel()", self.awq_layer.post_attention_layernorm.weight.numel())
        
        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.awq_layer.post_attention_layernorm.weight, self.model_config.rms_norm_eps)
        q = None
        k = None
        v = None

        # print("~"*18, "after FFN norm, LlamaTransformerLayer.forward:", '~'*18)
        # print("o", o, "o.shape", o.shape, "o.numel()", o.numel())
        # print("residual_buf", residual_buf, "residual_buf.shape", residual_buf.shape, "residual_buf.numel()", residual_buf.numel())

        # # Swift FFN Script:
        # print("~"*18, "before FFN, LlamaTransformerLayer.forward:", '~'*18)
        # print("weight: up_gate_proj", self.weight.up_gate_proj, "up_gate_proj.shape", self.weight.up_gate_proj.shape, "up_gate_proj.numel()", self.weight.up_gate_proj.numel())
        # print("weight: down_proj", self.weight.down_proj, "down_proj.shape", self.weight.down_proj.shape, "down_proj.numel()", self.weight.down_proj.numel())
        # up_gate_proj = linear(o, self.weight.up_gate_proj)
        # print("~"*18, "intermediate after o * up_gate_proj:", '~'*18)
        # print("up_gate_proj", up_gate_proj, "up_gate_proj.shape", up_gate_proj.shape, "up_gate_proj.numel()", up_gate_proj.numel())
        # silu_and_mul_inplace(up_gate_proj)
        # print("~"*18, "intermediate after silu_and_mul_inplace:", '~'*18)
        # print("up_gate_proj", up_gate_proj, "up_gate_proj.shape", up_gate_proj.shape, "up_gate_proj.numel()", up_gate_proj.numel())
        # ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        # print("~"*18, "final ffn_out:", '~'*18)
        # print("ffn_out", ffn_out, "ffn_out.shape", ffn_out.shape, "ffn_out.numel()", ffn_out.numel())
        # return ffn_out

        # Apply MLP/FFN using AWQ layers
        # print("~"*18, "before FFN, LlamaTransformerLayer.forward:", '~'*18)
        # print("gate_proj", self.awq_layer.mlp.gate_proj, "vars(self.awq_layer.mlp.gate_proj)", vars(self.awq_layer.mlp.gate_proj))
        # print("up_proj", self.awq_layer.mlp.up_proj, "vars(self.awq_layer.mlp.up_proj)", vars(self.awq_layer.mlp.up_proj))
        # print("down_proj", self.awq_layer.mlp.down_proj, "vars(self.awq_layer.mlp.down_proj)", vars(self.awq_layer.mlp.down_proj))
        # Reshape for AWQ
        reshaped_o = o.view(batch_size, seq_len, hidden_size)
        # print("~"*18, "reshaped_o", '~'*18)
        # print("reshaped_o", reshaped_o, "reshaped_o.shape", reshaped_o.shape, "reshaped_o.numel()", reshaped_o.numel())
        # Apply gate and up projections
        gate_proj = self.awq_layer.mlp.gate_proj(reshaped_o)
        # print("~"*18, "intermediate after o* gate_proj", '~'*18)
        # print("gate_proj", gate_proj, "gate_proj.shape", gate_proj.shape, "gate_proj.numel()", gate_proj.numel())
        up_proj = self.awq_layer.mlp.up_proj(reshaped_o)
        # print("~"*18, "intermediate after o * up_proj", '~'*18)
        # print("up_proj", up_proj, "up_proj.shape", up_proj.shape, "up_proj.numel()", up_proj.numel())
        # Apply SiLU activation and multiply
        gate_proj = torch.nn.functional.silu(gate_proj)
        # print("~"*18, "intermediate after silu(gate_proj)", '~'*18)
        # print("gate_proj", gate_proj, "gate_proj.shape", gate_proj.shape, "gate_proj.numel()", gate_proj.numel())
        ffn_output = gate_proj * up_proj
        # print("~"*18, "intermediate after gate_proj * up_proj", '~'*18)
        # print("ffn_output", ffn_output, "ffn_output.shape", ffn_output.shape, "ffn_output.numel()", ffn_output.numel())
        # Apply down projection
        ffn_output = self.awq_layer.mlp.down_proj(ffn_output)
        # print("~"*18, "intermediate after down_proj", '~'*18)
        # print("ffn_output", ffn_output, "ffn_output.shape", ffn_output.shape, "ffn_output.numel()", ffn_output.numel())
        # Reshape back to SwiftLLM format
        ffn_output = ffn_output.view(seq_len, hidden_size)
        # print("~"*18, "final ffn_out:", '~'*18)
        # print("ffn_out", ffn_output, "ffn_out.shape", ffn_output.shape, "ffn_out.numel()", ffn_output.numel())
        
        return ffn_output

         
         
