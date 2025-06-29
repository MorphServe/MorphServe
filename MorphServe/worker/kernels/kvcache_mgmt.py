import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.utils import cdiv

from typing import List

@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    my_batch_id = tl.program_id(0)
    my_block_id = tl.program_id(1)
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    if my_block_id*block_size >= my_seq_len:
        return
    
    my_token_range = tl.arange(0, block_size).to(tl.int64) + my_block_id*block_size + my_seq_start_loc
    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = (my_token_range*num_kv_heads*head_dim).to(tl.int64)[:, None, None] + (tl.arange(0, num_kv_heads)*head_dim)[None, :, None] + tl.arange(0, head_dim)[None, None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + \
        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
        tl.arange(0, head_dim)[None, None, :]
    
    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv, mask=mask), mask=mask)
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv, mask=mask), mask=mask)

@triton.jit
def _fwd_kvcache_mgmt_decoding_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    decoding_seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    decoding_seq_lens: torch.Tensor,  # [num_decoding_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_seq_id = tl.load(decoding_seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_block_id = (my_seq_len-1) // block_size
    my_block_offset = (my_seq_len-1) % block_size
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = my_batch_id*num_kv_heads*head_dim + (tl.arange(0, num_kv_heads)*head_dim)[:, None] + tl.arange(0, head_dim)[None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None] + my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]

    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv))
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv))


@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel_new_kv_combined(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    k_cache_new: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache_new: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    num_blocks_org: tl.constexpr,
    kv_cache_new_block_size: tl.constexpr,
    org_layer_param_size: tl.constexpr,
):
    # Notice: 
    # 1) due to the my_token_range is not influenced by the different block_tables (original or new), 
    #    the offs_kv is not influenced by the different block_tables also,
    #    it is only determined by the my_block_id and my_seq_start_loc, which is consistent for the original and new kv caches.
    # 2) the offs_kvcache is influenced by the different block_tables, 
    #    because the my_block_index is different from the original and new kv caches,
    #    but we correctly handle it with different kernels, and make the invalid block index with -1.
    # So, this kernel is the same for the original and new kv caches.

    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    # numth of the prefill sequence
    my_batch_id = tl.program_id(0)
    # numth of the block in the sequence
    my_block_id = tl.program_id(1)
    # prefill sequence token length
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    # prefill sequence token start location
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    # if the block is out of range: my_block_id*block_size >= my_seq_len, return
    if my_block_id*block_size >= my_seq_len:
        return
    
    # token range for k and v tensors that will be stored in the block
    # my_seq_start_loc is the start token of the sequence of the K and V tensors
    # my_block_id*block_size skips the previous blocks, which handles in the previous kernels < tl.program_id(1)
    my_token_range = tl.arange(0, block_size).to(tl.int64) + my_block_id*block_size + my_seq_start_loc
    # numth of the sequence
    my_seq_id = tl.load(seq_ids + my_batch_id)
    # numth of the block in the sequence in the block table
    # block_table is an address, and my_seq_id*max_blocks_per_seq to find the start of the sequence
    # then add my_block_id to find the block number
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    # Offs_kv: offs_kv.shape = [16, 32, 128] for the K and V tensors
    # (May needed to handle carefully to map the correct KV to new KV cache list)
    #   (my_token_range * 4096)[:, None, None] → shape [16, 1, 1]
    #   (head_id * 128)[None, :, None] → shape [1, 32, 1]
    #   (dim_id)[None, None, :] → shape [1, 1, 128]
    # When you broadcast and add these three:
    # Each element offs_kv[t, h, d] corresponds to the flattened offset in the 1D buffer for:
    #   token t, kv head h, dimension d
    offs_kv = (my_token_range*num_kv_heads*head_dim).to(tl.int64)[:, None, None] + \
        (tl.arange(0, num_kv_heads)*head_dim)[None, :, None] + \
        tl.arange(0, head_dim)[None, None, :]
    # offset of the key and value in the kvcache
    #   base_offset = (my_block_index * num_layers + cur_layer) * num_kv_heads * block_size * head_dim 
    #   skips the block and the layer. 
    # (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] is the head offset of the key and value in the kvcache
    # (tl.arange(0, block_size)*head_dim)[:, None, None] is the token offset of the key and value in the kvcache
    # tl.arange(0, head_dim)[None, None, :] is the dimension offset of the key and value in the kvcache
    #   For example:
    #   k_cache.shape = [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    #   block_size = 2
    #   head_dim = 4
    #   num_kv_heads = 2
    #   Then shape: [2, 2, 4] which is the same as offs_kv.shape
    #   offs_kvcache in detail:
    #       [[[base +  0, base +  1, base +  2, base +  3],   # token 0, head 0
    #       [base +  8, base +  9, base + 10, base + 11]],  # token 0, head 1
    #       [[base +  4, base +  5, base +  6, base +  7],   # token 1, head 0
    #       [base + 12, base + 13, base + 14, base + 15]]]  # token 1, head 1
    
    if my_block_index < num_blocks_org:
        offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + \
                        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
                        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
                        tl.arange(0, head_dim)[None, None, :]
        k_cache_ptrs = k_cache + offs_kvcache
        v_cache_ptrs = v_cache + offs_kvcache
    else:
        layer_skip = (my_block_index - num_blocks_org) // kv_cache_new_block_size
        offs_kvcache = ((my_block_index - num_blocks_org - layer_skip * kv_cache_new_block_size) * num_layers + cur_layer)*num_kv_heads*block_size*head_dim + \
                        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
                        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
                        tl.arange(0, head_dim)[None, None, :]
        k_cache_ptrs = k_cache_new - (layer_skip * org_layer_param_size) + offs_kvcache
        v_cache_ptrs = v_cache_new - (layer_skip * org_layer_param_size) + offs_kvcache

    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(k_cache_ptrs, tl.load(k + offs_kv, mask=mask), mask=mask)
    tl.store(v_cache_ptrs, tl.load(v + offs_kv, mask=mask), mask=mask)

@triton.jit
def _fwd_kvcache_mgmt_decoding_kernel_new_kv_combined(
    k_cache: torch.Tensor,	          # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	          # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	              # [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	              # [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    k_cache_new: torch.Tensor,	      # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache_new: torch.Tensor,	      # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,        # [max_seqs_in_block_table, max_blocks_per_seq], contiguous
    decoding_seq_ids: torch.Tensor,   # [num_decoding_seqs], contiguous
    decoding_seq_lens: torch.Tensor,  # [num_decoding_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    num_blocks_org: tl.constexpr,
    kv_cache_new_block_size: tl.constexpr,
    org_layer_param_size: tl.constexpr,
):
    # grid shape: [num_decoding_seqs]
    # numth of the decoding batch
    my_batch_id = tl.program_id(0).to(tl.int64)
    # numth of the sequence in the decoding batch
    my_seq_id = tl.load(decoding_seq_ids + my_batch_id)
    # length of the sequence in the decoding batch
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # numth of the block in the sequence in the block table
    my_block_id = (my_seq_len-1) // block_size
    # numth of the token in the my_block_id block in the block table
    my_block_offset = (my_seq_len-1) % block_size
    # the block_id (may be -1 or exceed the original kv cache size) corresponding to the block in the block table
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    # Offs_kv: offs_kv.shape = [32, 128] for the K and V tensors
    # The offs_kv is the offset of the KV in the K and V tensors, 
    # which is only related to the request of the same sequence, 
    # so it is the same for the original and new kv caches.
    #   (my_batch_id * 32 * 128) → base offset
    #   (head_id * 128)[None, :, None] → head offset [32, 1]
    #   (dim_id)[None, None, :] → dim offset [1, 128]
    offs_kv = my_batch_id*num_kv_heads*head_dim + (tl.arange(0, num_kv_heads)*head_dim)[:, None] + tl.arange(0, head_dim)[None, :]
    # offs_kvcache: offs_kvcache.shape = [32, 128] for the K and V cache tensors
    # The offs_kvcache is the offset of the KV in the K and V cache tensors, 
    # which is related to the block index on which block_table, 
    # so it is different for the original and new kv caches.
    # Base offset: (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim
    # Head offset: (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None]
    # Dim offset: my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]
    if my_block_index < num_blocks_org:
        offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None] + my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]
        k_cache_ptrs = k_cache + offs_kvcache
        v_cache_ptrs = v_cache + offs_kvcache
    else:
        layer_skip = (my_block_index - num_blocks_org) // kv_cache_new_block_size
        offs_kvcache = ((my_block_index - num_blocks_org - layer_skip * kv_cache_new_block_size) * num_layers + cur_layer)*num_kv_heads*block_size*head_dim + (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None] + my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]
        k_cache_ptrs = k_cache_new - (layer_skip * org_layer_param_size) + offs_kvcache
        v_cache_ptrs = v_cache_new - (layer_skip * org_layer_param_size) + offs_kvcache

    tl.store(k_cache_ptrs, tl.load(k + offs_kv))
    tl.store(v_cache_ptrs, tl.load(v + offs_kv))


def store_kvcache_org_version(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_ids.is_contiguous()
    assert infer_state.decoding_seq_lens.is_contiguous()

    if infer_state.num_prefill_seqs > 0:
        grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
        _fwd_kvcache_mgmt_prefill_kernel[grid](
            k_cache, v_cache,
            k, v,
            block_table,
            infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )

    if infer_state.num_decoding_seqs > 0:
        grid = (infer_state.num_decoding_seqs,)
        _fwd_kvcache_mgmt_decoding_kernel[grid](
            k_cache, v_cache,
            k[infer_state.num_prefill_tokens:, :, :],
            v[infer_state.num_prefill_tokens:, :, :],
            block_table,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.decoding_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )


def store_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_cache_new: List[torch.Tensor],
    v_cache_new: List[torch.Tensor],
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_ids.is_contiguous()
    assert infer_state.decoding_seq_lens.is_contiguous()

    def _launch(kc, vc, bt):
        """helper to re‐invoke your Triton kernels on (kc,vc) with block-table bt"""
        if infer_state.num_prefill_seqs > 0:
            grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
            _fwd_kvcache_mgmt_prefill_kernel[grid](
                kc, vc,
                k, v,
                bt,
                infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
                cur_layer,
                model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
            )

        if infer_state.num_decoding_seqs > 0:
            grid = (infer_state.num_decoding_seqs,)
            _fwd_kvcache_mgmt_decoding_kernel[grid](
                kc, vc,
                k[infer_state.num_prefill_tokens:, :, :],
                v[infer_state.num_prefill_tokens:, :, :],
                block_table,
                infer_state.seq_ids[infer_state.num_prefill_seqs:],
                infer_state.decoding_seq_lens,
                cur_layer,
                model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
            )
    
    def _launch_new_kv_combined(kc, vc, bt, kc_new, vc_new):
        """helper to re‐invoke your Triton kernels on (kc,vc) with block-table bt"""
        if infer_state.num_prefill_seqs > 0:
            grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
            _fwd_kvcache_mgmt_prefill_kernel_new_kv_combined[grid](
                kc, vc,
                k, v,
                kc_new, vc_new,
                bt,
                infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
                cur_layer,
                model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq,
                infer_state.num_blocks_org, infer_state.kv_cache_new_block_size, infer_state.org_layer_param_size
            )

        if infer_state.num_decoding_seqs > 0:
            grid = (infer_state.num_decoding_seqs,)
            _fwd_kvcache_mgmt_decoding_kernel_new_kv_combined[grid](
                kc, vc,
                k[infer_state.num_prefill_tokens:, :, :],
                v[infer_state.num_prefill_tokens:, :, :],
                kc_new, vc_new,
                bt,
                infer_state.seq_ids[infer_state.num_prefill_seqs:],
                infer_state.decoding_seq_lens,
                cur_layer,
                model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq,
                infer_state.num_blocks_org, infer_state.kv_cache_new_block_size, infer_state.org_layer_param_size
            )

    # 0-1) if there are no new kv caches, use the original kv cache
    num_new_groups = len(k_cache_new)
    if num_new_groups == 0:
        _launch(kc=k_cache, vc=v_cache, bt=block_table)
    else:
        # 0-2) if there are new kv caches, but all of them are not used, use the original kv cache
        num_blocks_org = infer_state.num_blocks_org
        mask0 = (block_table >= num_blocks_org)
        if not mask0.any():
            _launch(kc=k_cache, vc=v_cache, bt=block_table)
        else:

            # 1) there are new kv caches, and some of them are used;
            # block out the blocks exceed the original kv cache size, and store the original kv cache
            # After this step, all KV & tokens within the original kv cache size are stored in the original kv cache, 
            # and the tokens for the blocks exceed the original kv cache size are not stored yet.

            # print(f"\n1) ~~~~ store_kvcache for layer {cur_layer}, block out the blocks exceed the original kv cache size, and store the original kv cache with num_blocks_org: {num_blocks_org} ~~~~~~")
            # print(f"~~~~ k_cache shape: {k_cache.shape}")
            # print(f"~~~~ k_cache_new[0] shape: {k_cache_new[0].shape}")
            # print(f"~~~~ k shape: {k.shape}")
            # print(f"~~~~ block_table shape: {block_table.shape}, block_table: {block_table}")
            # print(f"infer_state.seq_ids: {infer_state.seq_ids}")
            # print(f"infer_state.decoding_seq_lens: {infer_state.decoding_seq_lens}")
            # print(f"infer_state.prefill_seq_lens: {infer_state.prefill_seq_lens}")
            # print(f"infer_state.prefill_seq_start_locs: {infer_state.prefill_seq_start_locs}")
            # print(f"infer_state.num_prefill_seqs: {infer_state.num_prefill_seqs}")
            # print(f"infer_state.num_prefill_tokens: {infer_state.num_prefill_tokens}")
            # print(f"infer_state.num_decoding_seqs: {infer_state.num_decoding_seqs}")
            # print(f"infer_state.max_prefill_len: {infer_state.max_prefill_len}")
            # print(f"~~~~ mask0 shape: {mask0.shape}, mask0: {mask0}")
            # print(f"~~~~ before block out ~~~~~~")
            # index_list = []
            # for index in range(block_table.shape[0]):
            #     if num_blocks_org <= max(block_table[index]):
            #         index_list.append(index)
            #         print(f"~~~index: {index} ~~~~")
            #         print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}]: {block_table[index]}")
            
            _launch_new_kv_combined(kc=k_cache, vc=v_cache, bt=block_table, kc_new=k_cache_new[0], vc_new=v_cache_new[0])
    
            '''
            # Done: verify the original kv cache store works correctly for decoding phase
            # -1 cannot be used as the block index during the decoding storing KV cache phase
            bt_org = block_table.clone()
            bt_org[mask0] = -1
            # print(f"\n~~~~ after block out ~~~~~~")
            # for index in index_list:
            #     print(f"~~~index: {index} ~~~~")
            #     print(f"bt_org[{index}].shape: {bt_org[index].shape}, bt_org[{index}]: {bt_org[index]}")
            _launch(kc=k_cache, vc=v_cache, bt=bt_org)

            # 2) for each new cache chunk j, isolate exactly those IDs and
            #    subtract the global offset to remap into [0..new_sizes[j]):
            # print(f"\n2) ~~~~~~ store_kvcache for layer {cur_layer}, for each new cache chunk j, isolate exactly those IDs and subtract the global offset to remap into [0..new_sizes[j]): ~~~~~~")
            kv_cache_new_block_size   = infer_state.kv_cache_new_block_size
            for j in range(num_new_groups):
                start = num_blocks_org + j*kv_cache_new_block_size
                end = start + kv_cache_new_block_size
                
                # build a block‐table that only contains IDs in [start,end):
                mask = (block_table < start) | (end <= block_table)
                if mask.all():
                    # print(f"store_kvcache: No blocks in the {j}th new kv cache, start: {start}, end: {end}")
                    continue
                
                btj = block_table.clone()

                # if infer_state.num_decoding_seqs > 0:
                #     print(f"\n~~~~~~~~ {j}th new kv cache, block_table, mask and btj details ~~~~~~~~")
                #     for index in range(block_table.shape[0]):
                #         if start <= max(block_table[index]) and max(block_table[index]) < end:
                #             print(f"~~~index: {index} ~~~~")
                #             print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}]: {block_table[index]}")
                #             print(f"mask[{index}].shape: {mask[index].shape}, mask[index]: {mask[index]}")
                
                btj = btj - start
                btj[mask] = -1
                
                # Done: verify the decoding KV store works correctly
                # if infer_state.num_decoding_seqs > 0:
                #     print("\n~~~~~ infer_state.num_decoding_seqs > 0 ~~~~~~")
                #     print("~"*18, f"store_kvcache for layer {cur_layer}, New blocks in the {j}th new kv cache, start: {start}, end: {end}", "~"*18)
                #     print(f"k_cache shape: {k_cache.shape}")
                #     print(f"k_cache_new shape: {k_cache_new[j].shape}")
                #     print(f"k shape: {k.shape}")
                #     print(f"btj shape: {btj.shape}")
                #     print(f"infer_state.seq_ids: {infer_state.seq_ids}")
                #     print(f"infer_state.decoding_seq_lens: {infer_state.decoding_seq_lens}")
                #     print(f"infer_state.prefill_seq_lens: {infer_state.prefill_seq_lens}")
                #     print(f"infer_state.prefill_seq_start_locs: {infer_state.prefill_seq_start_locs}")
                #     print(f"infer_state.num_prefill_seqs: {infer_state.num_prefill_seqs}")
                #     print(f"infer_state.num_prefill_tokens: {infer_state.num_prefill_tokens}")
                #     print(f"infer_state.num_decoding_seqs: {infer_state.num_decoding_seqs}")
                #     print(f"infer_state.max_prefill_len: {infer_state.max_prefill_len}\n")
                #     print(f"\n~~~~~~~~ {j}th new kv cache, btj after subtracting start ~~~~~~~~")
                #     for index in range(block_table.shape[0]):
                #         if start <= max(block_table[index]) and max(block_table[index]) < end:
                #             print(f"~~~index: {index} ~~~~")
                #             print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}]: {block_table[index]}")
                #             print(f"btj[{index}].shape: {btj[index].shape}, btj[{index}]: {btj[index]}")
                    # exit()
                _launch(kc=k_cache_new[j], vc=v_cache_new[j], bt=btj)
            '''


def store_kvcache_performance_test(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_cache_new: List[torch.Tensor],
    v_cache_new: List[torch.Tensor],
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_ids.is_contiguous()
    assert infer_state.decoding_seq_lens.is_contiguous()

    if infer_state.num_prefill_seqs > 0:
        grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
        _fwd_kvcache_mgmt_prefill_kernel[grid](
            k_cache, v_cache,
            k, v,
            block_table,
            infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )

    if infer_state.num_decoding_seqs > 0:
        grid = (infer_state.num_decoding_seqs,)
        _fwd_kvcache_mgmt_decoding_kernel[grid](
            k_cache, v_cache,
            k[infer_state.num_prefill_tokens:, :, :],
            v[infer_state.num_prefill_tokens:, :, :],
            block_table,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.decoding_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )
