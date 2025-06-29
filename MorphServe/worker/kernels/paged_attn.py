import time
import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState


'''
Using an example to explain the paged attention:

0) Varialble Setting:
    - One query token, shape: q = [1, 32, 128] → 1 token, 32 heads, 128-dimensional per head
    - Key/value cache blocks for seq_id = 0 (e.g., from block_table[0] = [2, 3, -1, -1, ...])
    - Total past sequence length = 21 tokens
    - Each block holds 16 tokens, so:
        - Block 2 has token indices [0–15]
        - Block 3 has token indices [16–20]
    - So the retrieved K/V values are for 21 tokens, across 32 heads, each of 128 dimensions.

1) Inputs:
    - q: query token, shape: [1, 32, 128]
        - q_0: [128],
        - q_1: [128],
        - ...
        - q_31: [128]
    - Cached key/value blocks, shape: [21, 32, 128]
    - Each key and value is per-token per-head, so we think of this as:
        - For token t:
            - K[t]: [32, 128]
            - V[t]: [32, 128]
        - But attention is head-wise. So we want to transpose to compute dot-product per head, 
          so the head would be the last dimension.
        
2) Reshape for Attention Computation
    - q = [32, 128]
    - K_reshaped = [32, 21, 128]  # 32 heads, 21 tokens, 128 dims
    - V_reshaped = [32, 21, 128]

3) Compute Attention Scores
    - For each head h = 0..31
        - q_h = [128]
        - K_reshaped[h] shape: [21, 128]
        - scores[h] = q[h] @ K_reshaped[h].T    # [21]
        - weights[h] = softmax(scores[h] / sqrt(128))  # [21]
    - So now attention weights: [32, 21] → 32 heads × 21 key tokens

4) Weighted Sum with Values
    -For each head h = 0..31
        - output[h] = weights[h] @ V_reshaped[h]  # [128]
    - So now output: [32, 128] → 32 heads × 128 dims
    - Flatten to [4096] → this becomes the final output vector for this token.

The final output vector for this token is [4096] → 32 heads × 128 dims
'''

@triton.jit
def _fwd_paged_attention_phase1(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous. num_seq_blocks = ceil(max_seq_len / seq_block_size)
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs, num_q_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: tl.float16,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_blocks: int,
    cur_layer: int,

    num_layers: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_my_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    
    # 1. Input Parameters and Index Setup
    my_batch_id = tl.program_id(0).to(tl.int64)     # which decoding sequence
    my_q_head_id = tl.program_id(1).to(tl.int64)    # which query head: 0, 1, 2, ..., 31
    my_seq_block_id = tl.program_id(2)              # which sequence block, what is a sequence block?
    
    # 2. Convert Q-head to K-head (GQA mapping)
    # In Grouped Query Attention (GQA), multiple query heads share one key/value head.
    #   - num_my_heads = num_q_heads // num_kv_heads
    #   - This maps each Q-head to its corresponding K/V head.
    #   - If you have 32 Q heads and 8 KV heads, each KV head handles 4 Q heads, such as Llama-3
    # But for Llama-2, the num_q_heads = 32, num_kv_heads = 32, 
    # so num_my_heads = num_my_heads = model_config.num_q_heads // model_config.num_kv_heads = 1
    # my_kv_head_id = 0, 1, 2, ..., 31
    my_kv_head_id = my_q_head_id // num_my_heads    

    # 3. Load metadata for this decoding sequence
    # my_seq_id = 0, 1, 2, ..., seq_id_max
    my_seq_id = tl.load(seq_ids + my_batch_id)
    # my_seq_len = len0, len1, len2, ..., len_seq_id_max, the length of the decoding sequence
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # my_start_token_idx = 0, 511, 1023, ..., seq_id_max*seq_block_size, with seq_block_size = 512 (hyper-parameter)
    # for example, If seq_len = 1100:
    #   - Block 0 → tokens [0…511]
    #   - Block 1 → tokens [512…1023]
    #   - Block 2 → tokens [1024…1099] ← not a full block, requires masking
    # We’re processing tokens [my_start_token_idx : my_start_token_idx + seq_block_size]
    my_start_token_idx = my_seq_block_id * seq_block_size

    # 4. Skip empty blocks
    # if the start token index is greater than the sequence length,
    # means that the there is no more tokens to process for this sequence block, return
    if my_start_token_idx >= my_seq_len:
        return
    
    # 5. Load the Query Vector for This Head
    # offs_q base address my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim
    # offs_q number is the head_dim
    offs_q = my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim + tl.arange(0, head_dim)
    # my_q is the query token for the current query head
    # my_q shape: [head_dim]
    my_q = tl.load(q + offs_q)  

    # 6. Prepare Block Offset and KV Pointer Calculation
    # seq_block_size = 512, block_size = 16, each kernel is responsible for processing 32 blocks
    # start_block_idx is the index of the first block id in block_table to process for the current sequence gpu block
    start_block_idx = my_seq_block_id*(seq_block_size//block_size)
    
    # k_cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # k_ptrs is the base pointer of the key cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # k_ptrs shape: [block_size, head_dim]
    # k_ptrs = k_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    k_ptrs = (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    
    # v_ptrs is the base pointer of the value cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # v_ptrs shape: [block_size, head_dim]
    # v_ptrs = v_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    v_ptrs = (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    
    # 7. Initialize Variables for Attention Calculation
    # max_score is the maximum score of the attention
    # Initialize the maximum attention score encountered so far (log-sum-exp trick needs this).
    # A very low starting point ensures the first real score will overwrite it.
    max_score = float("-1e20")
    # sum_exp is the sum of the exponential of the attention scores
    sum_exp = 0.0
    # Accumulator for the weighted sum of values
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # 8. Paged Attention Calculation
    # In the following code we deal with the case where the sequence block is
    # the last one in the sequence separately, because:
    #   - The last sequence block may not be a full block, therefore maskings
    #     are needed.
    #   - We can use tl.arange() when the sequence block is not the last one,
    #     leading to better performance.
    if my_start_token_idx + seq_block_size > my_seq_len:
        # The seq block I am processing is the last one in the sequence
        # Calculate how many 16-token blocks fit in the remainder.
        my_num_blocks = tl.cdiv(
            my_seq_len - my_start_token_idx,
            block_size
        )
        for block_i in range(0, my_num_blocks):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            k_block = tl.load(k_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
            attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
            v_block = tl.load(v_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score
    else:
        # The seq block I am processing is NOT the last one in the sequence
        # seq_block_size = 512 and block_size = 16 -> seq_block_size // block_size = 32
        for block_i in tl.static_range(0, seq_block_size // block_size):
            block_idx = start_block_idx + block_i
            # block_index is the actual (physical) memory block ID used by this sequence.
            # block_index could be -1, which means that the block is not used or index exceeds the original KV block number
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            
            # Retrieved from the block_table
            # based on the above k_ptrs, and we skip to the correct block index,we can get the key cache for the current block
            # k_block shape: [block_size, head_dim]
            k_block = tl.load(k_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + k_ptrs)
            
            # q⋅k(transposed): Dot product between query vector and each key vector in this block.
            # my_q shape: [128] — a query vector for one head
            # my_q[None, :] shape: [1, 128] — adds a batch dimension (broadcasting preparation)
            # k_block shape: [16, 128] — 16 keys in a block, each of dimension 128
            # atten_score shape: [16]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1)
            # originally: (q⋅k(transposed))* (1/sqrt(head_dim))
            # but here: (q⋅k(transposed))* (1/sqrt(head_dim)) * log2(e), which log2(e) = 1.442695040888963
            # to speed up the calculation, we use `exp2` instead of `exp` during the softmax calculation
            # Here we change softmax(q ⋅ kᵀ / sqrt(d)) → e**3 / (e**1+e**3+e**5+e**7) to 
            # exp2(3*log2(e)) / (exp2(1*log2(e))+exp2(3*log2(e))+exp2(5*log2(e))+exp2(7*log2(e)))
            attn_score = attn_score * softmax_scale
            # Load corresponding value vectors.
            # based on the above v_ptrs, and we skip to the correct block index,we can get the value cache for the current block
            # v_block shape: [16, 128]
            v_block = tl.load(v_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + v_ptrs)
            
            # Perform Numerically Stable Softmax
            # Find the max score in this block (to apply softmax stably).
            cur_max_score = tl.max(attn_score, axis=0)
            # Combine the old and new max score to track the global max seen so far.
            # This is the key to the log-sum-exp trick: update in log space without blowing up or underflowing.
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            # Shift scores to prevent overflow.
            # Exponentiate scores and adjust accumulated outputs accordingly.
            # Weighted sum of value vectors, softmax-style.
            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score

    # 9. Store Intermediate Outputs
    # such as [18, 32, 4, 128] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence, head_dim = 128
    offs_mid_o = my_batch_id*num_q_heads*num_seq_blocks*head_dim + (my_q_head_id*num_seq_blocks*head_dim) + my_seq_block_id*head_dim + tl.arange(0, head_dim)    
    tl.store(mid_o + offs_mid_o, acc / sum_exp)
    offs_mid_o_logexpsum = my_batch_id*num_q_heads*num_seq_blocks + my_q_head_id*num_seq_blocks + my_seq_block_id
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.math.log2(sum_exp) + max_score)   # Here tl.log(sum_exp) + max_score = log(sum(e^{a_i}))


@triton.jit
def _fwd_paged_attention_phase1_new_kv_combined(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous. num_seq_blocks = ceil(max_seq_len / seq_block_size)
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs, num_q_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: tl.float16,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_blocks: int,
    cur_layer: int,

    num_layers: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_my_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    num_blocks_org: tl.constexpr,
    org_layer_param_size: tl.constexpr,
    kv_cache_new_block_size: tl.constexpr,
    k_cache_new: torch.Tensor,
    v_cache_new: torch.Tensor,
):
    # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    
    # 1. Input Parameters and Index Setup
    my_batch_id = tl.program_id(0).to(tl.int64)     # which decoding sequence
    my_q_head_id = tl.program_id(1).to(tl.int64)    # which query head: 0, 1, 2, ..., 31
    my_seq_block_id = tl.program_id(2)              # which sequence block, what is a sequence block? a block is a sequence of tokens which is handled by one thread
    
    # 2. Convert Q-head to K-head (GQA mapping)
    # In Grouped Query Attention (GQA), multiple query heads share one key/value head.
    #   - num_my_heads = num_q_heads // num_kv_heads
    #   - This maps each Q-head to its corresponding K/V head.
    #   - If you have 32 Q heads and 8 KV heads, each KV head handles 4 Q heads, such as Llama-3
    # But for Llama-2, the num_q_heads = 32, num_kv_heads = 32, 
    # so num_my_heads = num_my_heads = model_config.num_q_heads // model_config.num_kv_heads = 1
    # my_kv_head_id = 0, 1, 2, ..., 31
    my_kv_head_id = my_q_head_id // num_my_heads    

    # 3. Load metadata for this decoding sequence
    # my_seq_id = 0, 1, 2, ..., seq_id_max
    my_seq_id = tl.load(seq_ids + my_batch_id)
    # my_seq_len = len0, len1, len2, ..., len_seq_id_max, the length of the decoding sequence
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # my_start_token_idx = 0, 511, 1023, ..., seq_id_max*seq_block_size, with seq_block_size = 512 (hyper-parameter)
    # for example, If seq_len = 1100:
    #   - Block 0 → tokens [0…511]
    #   - Block 1 → tokens [512…1023]
    #   - Block 2 → tokens [1024…1099] ← not a full block, requires masking
    # We’re processing tokens [my_start_token_idx : my_start_token_idx + seq_block_size]
    my_start_token_idx = my_seq_block_id * seq_block_size

    # 4. Skip empty blocks
    # if the start token index is greater than the sequence length,
    # means that the there is no more tokens to process for this sequence block, return
    if my_start_token_idx >= my_seq_len:
        return

    # 5. Load the Query Vector for This Head
    # offs_q base address my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim
    # offs_q number is the head_dim
    offs_q = my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim + tl.arange(0, head_dim)
    # my_q is the query token for the current query head
    # my_q shape: [head_dim]
    my_q = tl.load(q + offs_q)  

    # 6. Prepare Block Offset and KV Pointer Calculation
    # seq_block_size = 512, block_size = 16, each kernel is responsible for processing 32 blocks
    # start_block_idx is the index of the first block id in block_table to process for the current sequence gpu block
    start_block_idx = my_seq_block_id*(seq_block_size//block_size)
    
    # k_cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # k_ptrs is the base pointer of the key cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # k_ptrs shape: [block_size, head_dim]
    # k_ptrs = k_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    k_ptrs = (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    
    # v_ptrs is the base pointer of the value cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # v_ptrs shape: [block_size, head_dim]
    # v_ptrs = v_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    v_ptrs = (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]

    max_score = float("-1e20")
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # In the following code we deal with the case where the sequence block is
    # the last one in the sequence separately, because:
    #   - The last sequence block may not be a full block, therefore maskings
    #     are needed.
    #   - We can use tl.arange() when the sequence block is not the last one,
    #     leading to better performance.
    if my_start_token_idx + seq_block_size > my_seq_len:
        # The seq block I am processing is the last one in the sequence
        my_num_blocks = tl.cdiv(
            my_seq_len - my_start_token_idx,
            block_size
        )
        for block_i in range(0, my_num_blocks):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            if block_index < num_blocks_org:
                k_block = tl.load(k_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
                v_block = tl.load(v_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            else:
                layer_skip = (block_index - num_blocks_org) // kv_cache_new_block_size
                k_block = tl.load(k_cache_new - (layer_skip * org_layer_param_size) + (block_index - num_blocks_org - layer_skip * kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
                v_block = tl.load(v_cache_new - (layer_skip * org_layer_param_size) + (block_index - num_blocks_org - layer_skip * kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size:
            #     k_block = tl.load(k_cache_new_0 + (block_index - num_blocks_org)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_0 + (block_index - num_blocks_org)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*2:
            #     k_block = tl.load(k_cache_new_1 + (block_index - num_blocks_org - kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_1 + (block_index - num_blocks_org - kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*3:
            #     k_block = tl.load(k_cache_new_2 + (block_index - num_blocks_org - kv_cache_new_block_size*2)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_2 + (block_index - num_blocks_org - kv_cache_new_block_size*2)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*4:
            #     k_block = tl.load(k_cache_new_3 + (block_index - num_blocks_org - kv_cache_new_block_size*3)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_3 + (block_index - num_blocks_org - kv_cache_new_block_size*3)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*5:
            #     k_block = tl.load(k_cache_new_4 + (block_index - num_blocks_org - kv_cache_new_block_size*4)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_4 + (block_index - num_blocks_org - kv_cache_new_block_size*4)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # else:
            #     k_block = tl.load(k_cache_new_5 + (block_index - num_blocks_org - kv_cache_new_block_size*5)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_5 + (block_index - num_blocks_org - kv_cache_new_block_size*5)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
            attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score
    else:
        # The seq block I am processing is NOT the last one in the sequence
        for block_i in tl.static_range(0, seq_block_size // block_size):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            if block_index < num_blocks_org:
                k_block = tl.load(k_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
                v_block = tl.load(v_cache + block_index*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            else:
                layer_skip = (block_index - num_blocks_org) // kv_cache_new_block_size
                k_block = tl.load(k_cache_new - (layer_skip * org_layer_param_size) + (block_index - num_blocks_org - layer_skip * kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
                v_block = tl.load(v_cache_new - (layer_skip * org_layer_param_size) + (block_index - num_blocks_org - layer_skip * kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size:
            #     k_block = tl.load(k_cache_new_0 + (block_index - num_blocks_org)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_0 + (block_index - num_blocks_org)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*2:
            #     k_block = tl.load(k_cache_new_1 + (block_index - num_blocks_org - kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_1 + (block_index - num_blocks_org - kv_cache_new_block_size)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*3:
            #     k_block = tl.load(k_cache_new_2 + (block_index - num_blocks_org - kv_cache_new_block_size*2)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_2 + (block_index - num_blocks_org - kv_cache_new_block_size*2)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*4:
            #     k_block = tl.load(k_cache_new_3 + (block_index - num_blocks_org - kv_cache_new_block_size*3)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_3 + (block_index - num_blocks_org - kv_cache_new_block_size*3)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # elif block_index < num_blocks_org + kv_cache_new_block_size*5:
            #     k_block = tl.load(k_cache_new_4 + (block_index - num_blocks_org - kv_cache_new_block_size*4)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_4 + (block_index - num_blocks_org - kv_cache_new_block_size*4)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            # else:
            #     k_block = tl.load(k_cache_new_5 + (block_index - num_blocks_org - kv_cache_new_block_size*5)*num_layers*num_kv_heads*block_size*head_dim + k_ptrs) # [block_size, head_dim]
            #     v_block = tl.load(v_cache_new_5 + (block_index - num_blocks_org - kv_cache_new_block_size*5)*num_layers*num_kv_heads*block_size*head_dim + v_ptrs) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score

    offs_mid_o = my_batch_id*num_q_heads*num_seq_blocks*head_dim + my_seq_block_id*head_dim + (my_q_head_id*num_seq_blocks*head_dim) + tl.arange(0, head_dim)
    tl.store(mid_o + offs_mid_o, acc / sum_exp)
    offs_mid_o_logexpsum = my_batch_id*num_q_heads*num_seq_blocks + my_seq_block_id + my_q_head_id*num_seq_blocks
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.math.log2(sum_exp) + max_score)   # Here tl.log(sum_exp) + max_score = log(sum(e^{a_i}))



@triton.jit
def _fwd_paged_attention_phase1_new_kv_used(
    acc_buf,                # [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous
    sum_exp_buf,            # [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    max_score_buf,          # [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs, num_q_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: tl.float16,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_blocks: int,
    cur_layer: int,

    num_layers: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_my_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    
    # 1. Input Parameters and Index Setup
    my_batch_id = tl.program_id(0).to(tl.int64)     # which decoding sequence
    my_q_head_id = tl.program_id(1).to(tl.int64)    # which query head: 0, 1, 2, ..., 31
    my_seq_block_id = tl.program_id(2)              # which sequence block, what is a sequence block?
    
    # 2. Convert Q-head to K-head (GQA mapping)
    # In Grouped Query Attention (GQA), multiple query heads share one key/value head.
    #   - num_my_heads = num_q_heads // num_kv_heads
    #   - This maps each Q-head to its corresponding K/V head.
    #   - If you have 32 Q heads and 8 KV heads, each KV head handles 4 Q heads, such as Llama-3
    # But for Llama-2, the num_q_heads = 32, num_kv_heads = 32, 
    # so num_my_heads = num_my_heads = model_config.num_q_heads // model_config.num_kv_heads = 1
    # my_kv_head_id = 0, 1, 2, ..., 31
    my_kv_head_id = my_q_head_id // num_my_heads    

    # 3. Load metadata for this decoding sequence
    # my_seq_id = 0, 1, 2, ..., seq_id_max
    my_seq_id = tl.load(seq_ids + my_batch_id)
    # my_seq_len = len0, len1, len2, ..., len_seq_id_max, the length of the decoding sequence
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # my_start_token_idx = 0, 511, 1023, ..., seq_id_max*seq_block_size, with seq_block_size = 512 (hyper-parameter)
    # for example, If seq_len = 1100:
    #   - Block 0 → tokens [0…511]
    #   - Block 1 → tokens [512…1023]
    #   - Block 2 → tokens [1024…1099] ← not a full block, requires masking
    # We’re processing tokens [my_start_token_idx : my_start_token_idx + seq_block_size]
    my_start_token_idx = my_seq_block_id * seq_block_size

    # 4. Skip empty blocks
    # if the start token index is greater than the sequence length,
    # means that the there is no more tokens to process for this sequence block, return
    if my_start_token_idx >= my_seq_len:
        return
    
    # 5. Load the Query Vector for This Head
    # offs_q base address my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim
    # offs_q number is the head_dim
    offs_q = my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim + tl.arange(0, head_dim)
    # my_q is the query token for the current query head
    # my_q shape: [head_dim]
    my_q = tl.load(q + offs_q)  

    # 6. Prepare Block Offset and KV Pointer Calculation
    # seq_block_size = 512, block_size = 16, each kernel is responsible for processing 32 blocks
    # start_block_idx is the index of the first block id in block_table to process for the current sequence gpu block
    start_block_idx = my_seq_block_id*(seq_block_size//block_size)
    
    # k_cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # k_ptrs is the base pointer of the key cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # k_ptrs shape: [block_size, head_dim]
    k_ptrs = k_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    # v_ptrs is the base pointer of the value cache, 
    # relative to the current layer and KV head, 
    # and later we will have to add the block index to get the correct block pointer
    # v_ptrs shape: [block_size, head_dim]
    v_ptrs = v_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]

    # 7. Initialize Variables for Attention Calculation
    # max_score is the maximum score of the attention
    # Initialize the maximum attention score encountered so far (log-sum-exp trick needs this).
    # A very low starting point ensures the first real score will overwrite it.
    offs_max_score = my_batch_id*num_q_heads*num_seq_blocks + my_q_head_id*num_seq_blocks + my_seq_block_id
    max_score = tl.load(max_score_buf + offs_max_score)
    # sum_exp is the sum of the exponential of the attention scores
    offs_sum_exp = my_batch_id*num_q_heads*num_seq_blocks + my_q_head_id*num_seq_blocks + my_seq_block_id
    sum_exp = tl.load(sum_exp_buf + offs_sum_exp)
    # Accumulator for the weighted sum of values
    offs_acc = my_batch_id*num_q_heads*num_seq_blocks*head_dim + my_q_head_id*num_seq_blocks*head_dim + my_seq_block_id*head_dim + tl.arange(0, head_dim)
    acc = tl.load(acc_buf + offs_acc)

    # 8. Paged Attention Calculation
    # In the following code we deal with the case where the sequence block is
    # the last one in the sequence separately, because:
    #   - The last sequence block may not be a full block, therefore maskings
    #     are needed.
    #   - We can use tl.arange() when the sequence block is not the last one,
    #     leading to better performance.
    if my_start_token_idx + seq_block_size > my_seq_len:
        # The seq block I am processing is the last one in the sequence
        # Calculate how many 16-token blocks fit in the remainder.
        my_num_blocks = tl.cdiv(
            my_seq_len - my_start_token_idx,
            block_size
        )
        for block_i in range(0, my_num_blocks):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            # if block_index is -1, skip this block
            if block_index != -1:
                k_block = tl.load(k_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
                attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
                attn_score = attn_score * softmax_scale
                offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
                attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
                v_block = tl.load(v_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
                
                cur_max_score = tl.max(attn_score, axis=0)
                new_max_score = tl.maximum(max_score, cur_max_score)
                exp_attn_score = tl.math.exp2(attn_score - new_max_score)
                old_acc_scale = tl.math.exp2(max_score - new_max_score)

                acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
                sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
                max_score = new_max_score
    else:
        # The seq block I am processing is NOT the last one in the sequence
        # seq_block_size = 512 and block_size = 16 -> seq_block_size // block_size = 32
        for block_i in tl.static_range(0, seq_block_size // block_size):
            block_idx = start_block_idx + block_i
            # block_index is the actual (physical) memory block ID used by this sequence.
            # block_index could be -1, which means that the block is not used or index exceeds the original KV block number
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            # if block_index is -1, skip this block
            if block_index != -1:
                # Retrieved from the block_table
                # based on the above k_ptrs, and we skip to the correct block index,we can get the key cache for the current block
                # k_block shape: [block_size, head_dim]
                k_block = tl.load(k_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim)
                
                # q⋅k(transposed): Dot product between query vector and each key vector in this block.
                # my_q shape: [128] — a query vector for one head
                # my_q[None, :] shape: [1, 128] — adds a batch dimension (broadcasting preparation)
                # k_block shape: [16, 128] — 16 keys in a block, each of dimension 128
                # atten_score shape: [16]
                attn_score = tl.sum(my_q[None, :] * k_block, axis=1)
                # originally: (q⋅k(transposed))* (1/sqrt(head_dim))
                # but here: (q⋅k(transposed))* (1/sqrt(head_dim)) * log2(e), which log2(e) = 1.442695040888963
                # to speed up the calculation, we use `exp2` instead of `exp` during the softmax calculation
                # Here we change softmax(q ⋅ kᵀ / sqrt(d)) → e**3 / (e**1+e**3+e**5+e**7) to 
                # exp2(3*log2(e)) / (exp2(1*log2(e))+exp2(3*log2(e))+exp2(5*log2(e))+exp2(7*log2(e)))
                attn_score = attn_score * softmax_scale
                # Load corresponding value vectors.
                # based on the above v_ptrs, and we skip to the correct block index,we can get the value cache for the current block
                # v_block shape: [16, 128]
                v_block = tl.load(v_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim)
                
                # Perform Numerically Stable Softmax
                # Find the max score in this block (to apply softmax stably).
                cur_max_score = tl.max(attn_score, axis=0)
                # Combine the old and new max score to track the global max seen so far.
                # This is the key to the log-sum-exp trick: update in log space without blowing up or underflowing.
                new_max_score = tl.maximum(max_score, cur_max_score)
                exp_attn_score = tl.math.exp2(attn_score - new_max_score)
                old_acc_scale = tl.math.exp2(max_score - new_max_score)

                # Shift scores to prevent overflow.
                # Exponentiate scores and adjust accumulated outputs accordingly.
                # Weighted sum of value vectors, softmax-style.
                acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
                sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
                max_score = new_max_score

    # 9. Update the intermediate outputs
    tl.store(max_score_buf + offs_max_score, max_score)
    tl.store(sum_exp_buf + offs_sum_exp, sum_exp)
    tl.store(acc_buf + offs_acc, acc)


@triton.jit
def _paged_attention_mid_phase(
    acc_buf,             # [B, H, S, D]
    sum_exp_buf,         # [B, H, S]
    max_score_buf,       # [B, H, S]
    mid_o,               # [B, H, S, D]
    mid_o_logexpsum,     # [B, H, S]
    head_dim: tl.constexpr,
    num_seq_blocks: tl.constexpr,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    s_id = tl.program_id(2)

    # --- Index calculation ---
    offset_acc = ((b_id * head_dim * num_seq_blocks + s_id * head_dim) + h_id * num_seq_blocks * head_dim) + tl.arange(0, head_dim)
    offset_out = offset_acc
    offset_sum = (b_id * num_seq_blocks + s_id) + h_id * num_seq_blocks
    offset_logexp = offset_sum

    acc_val = tl.load(acc_buf + offset_acc).to(tl.float32)
    sum_exp_val = tl.load(sum_exp_buf + offset_sum).to(tl.float32)
    max_score_val = tl.load(max_score_buf + offset_sum).to(tl.float32)

    mid_o_val = acc_val / sum_exp_val
    logexpsum_val = tl.math.log2(sum_exp_val) + max_score_val

    tl.store(mid_o + offset_out, mid_o_val)
    tl.store(mid_o_logexpsum + offset_logexp, logexpsum_val)


@triton.jit
def _fwd_paged_attention_phase2(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    o: torch.Tensor,		# [num_decoding_seqs, num_q_heads, head_dim], contiguous

    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous

    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_seq_blocks: tl.constexpr,
    seq_block_size: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_q_heads]
    my_batch_id = tl.program_id(0)
    my_q_head_id = tl.program_id(1)

    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_num_seq_blocks = tl.cdiv(my_seq_len, seq_block_size)

    sum_exp = 0.0
    max_score = float("-1e20")
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for seq_block_id in range(my_num_seq_blocks):
        offs_mid_o = ((my_batch_id*num_q_heads+my_q_head_id)*num_seq_blocks+seq_block_id)*head_dim + tl.arange(0, head_dim)
        offs_mid_o_logexpsum = (my_batch_id*num_q_heads+my_q_head_id)*num_seq_blocks+seq_block_id
        cur_mid_o = tl.load(mid_o + offs_mid_o)   # [head_dim]
        cur_mid_o_logexpsum = tl.load(mid_o_logexpsum + offs_mid_o_logexpsum)

        new_max_score = tl.maximum(max_score, cur_mid_o_logexpsum)
        old_scale = tl.math.exp2(max_score - new_max_score)
        exp_score = tl.math.exp2(cur_mid_o_logexpsum - new_max_score)
        acc = acc * old_scale + exp_score * cur_mid_o
        sum_exp = sum_exp * old_scale + exp_score
        max_score = new_max_score

    offs_o = (my_batch_id*num_q_heads+my_q_head_id)*head_dim + tl.arange(0, head_dim)
    tl.store(o + offs_o, (acc / sum_exp).to(tl.float16))



def paged_attention(
    q: torch.Tensor,                    # [num_decoding_seqs, num_q_heads, head_dim]
    k_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    k_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    block_table: torch.Tensor,          # [max_seqs_in_block_table, max_blocks_per_seq]
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int,
    o: torch.Tensor                     # [num_decoding_seqs, num_q_heads, head_dim]
):
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_block_size % engine_config.block_size == 0
    assert o.is_contiguous()

    # Performance bottleneck could be happening at the following:
    # 1) Memory Allocation Bottleneck for mid_o, mid_o_logexpsum, o
    #    - Issue: mid_o, mid_o_logexpsum, o shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    #    - Solution: pre-allocate the memory for mid_o, mid_o_logexpsum, o
    # 2) Kernel Execution (Triton Kernel) Inefficiency or Stalls
    #    - Issue: Triton Kernel is not efficient or stalls:
    #          Not optimized for the current grid shape
    #          Has too many warps or shared memory
    #          Involves poor memory coalescing
    #    - Solution: try to fine tune the hyper-parameters for the grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # 3) Shape Explosion / Too Much Work
    #    - Issue: If the decoding q shape is [18, 32, 128], 
    #              but your infer_state.num_seq_blocks is huge (e.g., hundreds of blocks due to long sequences),
    #              This will launch a massive number of threads in Triton.
    #    - Solution: Check the size of infer_state.num_seq_blocks 
    #                — if it's large (say > 512), you might be over-parallelizing
    #                or doing excessive memory reads/writes
    # 4) Memory Coalescing Issues
    #    - Issue: Poor memory coalescing in the Triton kernel
    #    - Solution: Ensure the memory access pattern is coalesced
    # 5) Load Balancing
    #    - Issue: Uneven work distribution among threads

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
    
    
    # q = current query tokens for each decoding sequence
    # q shape: [num_decoding_seqs, num_q_heads, head_dim]
    # such as [18, 32, 128] | 18 decoding sequences, 32 attention heads, head_dim = 128

    # k_cache, v_cache = previously cached keys/values (from prefill + earlier decode steps)
    # k_cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # such as [1031, 32, 32, 16, 128] | 1031 blocks, 32 layers, 32 kv_heads, block_size=16, head_dim=128

    # k_cache_new, v_cache_new = new keys/values (from prefill + previous decode steps)

    # block_table = maps each sequence ID to the physical blocks of KV cache it owns
    # block_table shape: [max_seqs_in_block_table, max_blocks_per_seq]
    # such as [128, 512] | For each sequence, maps up to 512 logical blocks to physical block ids

    # o = output tokens of the paged attention
    # o shape: [num_decoding_seqs, layer_hidden_dimension (num_q_heads*head_dim)]
    # such as [18, 4096] | Output tokens — probably flattened, where 4096 = 32 heads × 128 dims

    start_time_paged_attention = time.perf_counter()
    # mid_o = intermediate attention scores for each decoding sequence
    # mid_o shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    # such as [18, 32, 4, 128] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence, head_dim = 128
    mid_o = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)

    # mid_o_logexpsum = log-sum-exp of the intermediate attention scores for each decoding sequence
    # mid_o_logexpsum shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    # # acc_buf = accumulated attention scores for each decoding sequence
    # # acc_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    # # such as [18, 32, 4, 128] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence, head_dim = 128
    # acc_buf = torch.zeros((
    #     infer_state.num_decoding_seqs,
    #     model_config.num_q_heads,
    #     infer_state.num_seq_blocks,
    #     model_config.head_dim
    # ), device=q.device, dtype=torch.float32)

    # # sum_exp_buf = sum of the exponential of the attention scores for each decoding sequence
    # # sum_exp_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
    # sum_exp_buf = torch.zeros((
    #     infer_state.num_decoding_seqs,
    #     model_config.num_q_heads,
    #     infer_state.num_seq_blocks
    # ), device=q.device, dtype=torch.float32)

    # # max_score_buf = maximum attention score for each decoding sequence
    # # max_score_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
    # max_score_buf = torch.full((
    #     infer_state.num_decoding_seqs,
    #     model_config.num_q_heads,
    #     infer_state.num_seq_blocks
    # ), fill_value=-1e20, device=q.device, dtype=torch.float32)

    end_time_intermediate_outputs_init = time.perf_counter()

    # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
    # Each thread in the grid is responsible for computing the attention output for:
    #   - a decoding sequence b
    #   - a query head h
    #   - a chunk of its history defined by my_seq_block_id
    def _launch_phase1(k_cache, v_cache, block_table):
        grid = (infer_state.num_decoding_seqs, model_config.num_q_heads, infer_state.num_seq_blocks)
        _fwd_paged_attention_phase1[grid](
            mid_o, mid_o_logexpsum,
            q, k_cache, v_cache,
            block_table,
            infer_state.softmax_scale * 1.442695040888963,
            infer_state.decoding_seq_lens,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.num_seq_blocks,
            cur_layer,

            model_config.num_layers,
            model_config.num_q_heads,
            model_config.num_kv_heads,
            model_config.num_q_heads // model_config.num_kv_heads,
            engine_config.block_size,
            model_config.head_dim,
            infer_state.seq_block_size,
            engine_config.max_blocks_per_seq,
            num_warps = 1,
            num_stages = 4
        )

    def _launch_phase1_new_kv_combined(k_cache, v_cache, block_table, 
                                       k_cache_new, v_cache_new):
        grid = (infer_state.num_decoding_seqs, model_config.num_q_heads, infer_state.num_seq_blocks)
        _fwd_paged_attention_phase1_new_kv_combined[grid](
            mid_o, mid_o_logexpsum,
            q, k_cache, v_cache,
            block_table,
            infer_state.softmax_scale * 1.442695040888963,
            infer_state.decoding_seq_lens,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.num_seq_blocks,
            cur_layer,
            model_config.num_layers,
            model_config.num_q_heads,
            model_config.num_kv_heads,
            model_config.num_q_heads // model_config.num_kv_heads,
            engine_config.block_size,
            model_config.head_dim,
            infer_state.seq_block_size,
            engine_config.max_blocks_per_seq,
            infer_state.num_blocks_org,
            infer_state.org_layer_param_size,
            infer_state.kv_cache_new_block_size,
            k_cache_new,
            v_cache_new,
            num_warps = 1,
            num_stages = 4
        )

    # 0-1) if there are no new kv caches, use the original kv cache
    num_new_groups = len(k_cache_new)
    if num_new_groups == 0:
        _launch_phase1(k_cache=k_cache, v_cache=v_cache, block_table=block_table)
        # _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
        #                                         block_table=block_table, 
        #                                         k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0])
    else:
        # 0-2) if there are new kv caches, but all of them are not used, use the original kv cache
        num_blocks_org = infer_state.num_blocks_org
        mask0 = (block_table >= num_blocks_org)
        if not mask0.any():
            # print(f"~~~~ paged_attention for layer {cur_layer}, all new kv caches are not used ~~~~")
            _launch_phase1(k_cache=k_cache, v_cache=v_cache, block_table=block_table)
            # _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0])
        else:
            # 1) there are new kv caches, and some of them are used;
            # print(f"!!!!!! paged_attention for layer {cur_layer}, some new kv caches are used !!!!!!")
            # print(f"~~~~ q shape: {q.shape}")
            # print(f"~~~~ k_cache shape: {k_cache.shape}")
            # print(f"~~~~ k_cache_new[0] shape: {k_cache_new[0].shape}")
            # print(f"~~~~ k_cache_new length: {len(k_cache_new)}")
            # print(f"~~~~ o shape: {o.shape}")
            # print(f"~~~~ block_table shape: {block_table.shape}, block_table: {block_table}")
            # print(f"infer_state.seq_ids: {infer_state.seq_ids}")
            # print(f"infer_state.num_decoding_seqs: {infer_state.num_decoding_seqs}")
            # print(f"infer_state.num_q_heads: {model_config.num_q_heads}")
            # print(f"infer_state.num_kv_heads: {model_config.num_kv_heads}")
            # print(f"infer_state.num_seq_blocks: {infer_state.num_seq_blocks}")
            # print(f"infer_state.seq_block_size: {infer_state.seq_block_size}")
            # print(f"infer_state.decoding_seq_lens: {infer_state.decoding_seq_lens}")
            # print(f"infer_state.max_blocks_per_seq: {engine_config.max_blocks_per_seq}")
            # print(f"infer_state.softmax_scale: {infer_state.softmax_scale}")
            # print(f"infer_state.prefill_seq_lens: {infer_state.prefill_seq_lens}")
            # print(f"infer_state.prefill_seq_start_locs: {infer_state.prefill_seq_start_locs}")
            # print(f"infer_state.num_prefill_seqs: {infer_state.num_prefill_seqs}")
            # print(f"infer_state.num_prefill_tokens: {infer_state.num_prefill_tokens}")
            # print(f"infer_state.max_prefill_len: {infer_state.max_prefill_len}")
            # print(f"~~~~ mask0 shape: {mask0.shape}, mask0: {mask0}")
            # print(f"~~~~ before block out ~~~~~~")
            # index_list = []
            # for index in range(block_table.shape[0]):
            #     if num_blocks_org <= max(block_table[index]):
            #         index_list.append(index)
            #         print(f"~~~index: {index} ~~~~")
            #         print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}] first 256: {block_table[index][:256]}")
            # exit()
            # mask1 = (block_table >= num_blocks_org+infer_state.kv_cache_new_block_size)
            # if not mask1.any():
            _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
                                          block_table=block_table, 
                                          k_cache_new=k_cache_new[0], v_cache_new=v_cache_new[0])
            # if len(k_cache_new) == 1:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=torch.zeros(1, device=k_cache_new[0].device), v_cache_new_1=torch.zeros(1, device=k_cache_new[0].device),
            #                                     k_cache_new_2=torch.zeros(1, device=k_cache_new[0].device), v_cache_new_2=torch.zeros(1, device=k_cache_new[0].device),
            #                                     k_cache_new_3=torch.zeros(1, device=k_cache_new[0].device), v_cache_new_3=torch.zeros(1, device=k_cache_new[0].device),
            #                                     k_cache_new_4=torch.zeros(1, device=k_cache_new[0].device), v_cache_new_4=torch.zeros(1, device=k_cache_new[0].device),
            #                                     k_cache_new_5=torch.zeros(1, device=k_cache_new[0].device), v_cache_new_5=torch.zeros(1, device=k_cache_new[0].device))
            # elif len(k_cache_new) == 2:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=k_cache_new[1], v_cache_new_1=v_cache_new[1],
            #                                     k_cache_new_2=None, v_cache_new_2=None,
            #                                     k_cache_new_3=None, v_cache_new_3=None,
            #                                     k_cache_new_4=None, v_cache_new_4=None,
            #                                     k_cache_new_5=None, v_cache_new_5=None)
            # elif len(k_cache_new) == 3:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=k_cache_new[1], v_cache_new_1=v_cache_new[1],
            #                                     k_cache_new_2=k_cache_new[2], v_cache_new_2=v_cache_new[2],
            #                                     k_cache_new_3=None, v_cache_new_3=None,
            #                                     k_cache_new_4=None, v_cache_new_4=None,
            #                                     k_cache_new_5=None, v_cache_new_5=None)
            # elif len(k_cache_new) == 4:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=k_cache_new[1], v_cache_new_1=v_cache_new[1],
            #                                     k_cache_new_2=k_cache_new[2], v_cache_new_2=v_cache_new[2],
            #                                     k_cache_new_3=k_cache_new[3], v_cache_new_3=v_cache_new[3],
            #                                     k_cache_new_4=None, v_cache_new_4=None,
            #                                     k_cache_new_5=None, v_cache_new_5=None)
            # elif len(k_cache_new) == 5:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=k_cache_new[1], v_cache_new_1=v_cache_new[1],
            #                                     k_cache_new_2=k_cache_new[2], v_cache_new_2=v_cache_new[2],
            #                                     k_cache_new_3=k_cache_new[3], v_cache_new_3=v_cache_new[3],
            #                                     k_cache_new_4=k_cache_new[4], v_cache_new_4=v_cache_new[4],
            #                                     k_cache_new_5=None, v_cache_new_5=None)
            # elif len(k_cache_new) == 6:
            #     _launch_phase1_new_kv_combined(k_cache=k_cache, v_cache=v_cache, 
            #                                     block_table=block_table, 
            #                                     k_cache_new_0=k_cache_new[0], v_cache_new_0=v_cache_new[0],
            #                                     k_cache_new_1=k_cache_new[1], v_cache_new_1=v_cache_new[1],
            #                                     k_cache_new_2=k_cache_new[2], v_cache_new_2=v_cache_new[2],
            #                                     k_cache_new_3=k_cache_new[3], v_cache_new_3=v_cache_new[3],
            #                                     k_cache_new_4=k_cache_new[4], v_cache_new_4=v_cache_new[4],
            #                                     k_cache_new_5=k_cache_new[5], v_cache_new_5=v_cache_new[5])
            # else:
            #     raise ValueError(f"len(k_cache_new) = {len(k_cache_new)} is not supported")
        
    end_time_paged_attention_phase1 = time.perf_counter()

    grid = (infer_state.num_decoding_seqs, model_config.num_q_heads)
    _fwd_paged_attention_phase2[grid](
        mid_o, mid_o_logexpsum,
        o,
        infer_state.decoding_seq_lens,
        model_config.num_q_heads,
        model_config.head_dim,
        infer_state.num_seq_blocks,
        infer_state.seq_block_size,
    )
    end_time_paged_attention_phase2 = time.perf_counter()

    time_threshold = 0.5 # 500 ms
    if (end_time_paged_attention_phase2 - start_time_paged_attention) > time_threshold:
        print(f"\n[PagedAttention-Serving] PagedAttention serving time exceeded {time_threshold} s with {(end_time_paged_attention_phase2 - start_time_paged_attention):.2f} s.")
        print(f"[PagedAttention-Serving] Time to intermediate_outputs_init: {(end_time_intermediate_outputs_init - start_time_paged_attention)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Time to paged_attention_phase1:    {(end_time_paged_attention_phase1 - end_time_intermediate_outputs_init)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Time to paged_attention_phase2:    {(end_time_paged_attention_phase2 - end_time_paged_attention_phase1)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Total time: {(end_time_paged_attention_phase2 - start_time_paged_attention)*1000:.2f} ms.")



# this version is used for handle original and new kv cache separately, with multiple kernels
def paged_attention_multi_kernels(
    q: torch.Tensor,                    # [num_decoding_seqs, num_q_heads, head_dim]
    k_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    k_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    block_table: torch.Tensor,          # [max_seqs_in_block_table, max_blocks_per_seq]
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int,
    o: torch.Tensor                     # [num_decoding_seqs, num_q_heads, head_dim]
):
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_block_size % engine_config.block_size == 0
    assert o.is_contiguous()

    # Performance bottleneck could be happening at the following:
    # 1) Memory Allocation Bottleneck for mid_o, mid_o_logexpsum, o
    #    - Issue: mid_o, mid_o_logexpsum, o shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    #    - Solution: pre-allocate the memory for mid_o, mid_o_logexpsum, o
    # 2) Kernel Execution (Triton Kernel) Inefficiency or Stalls
    #    - Issue: Triton Kernel is not efficient or stalls:
    #          Not optimized for the current grid shape
    #          Has too many warps or shared memory
    #          Involves poor memory coalescing
    #    - Solution: try to fine tune the hyper-parameters for the grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # 3) Shape Explosion / Too Much Work
    #    - Issue: If the decoding q shape is [18, 32, 128], 
    #              but your infer_state.num_seq_blocks is huge (e.g., hundreds of blocks due to long sequences),
    #              This will launch a massive number of threads in Triton.
    #    - Solution: Check the size of infer_state.num_seq_blocks 
    #                — if it's large (say > 512), you might be over-parallelizing
    #                or doing excessive memory reads/writes
    # 4) Memory Coalescing Issues
    #    - Issue: Poor memory coalescing in the Triton kernel
    #    - Solution: Ensure the memory access pattern is coalesced
    # 5) Load Balancing
    #    - Issue: Uneven work distribution among threads

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
    
    
    # q = current query tokens for each decoding sequence
    # q shape: [num_decoding_seqs, num_q_heads, head_dim]
    # such as [18, 32, 128] | 18 decoding sequences, 32 attention heads, head_dim = 128

    # k_cache, v_cache = previously cached keys/values (from prefill + earlier decode steps)
    # k_cache shape: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # such as [1031, 32, 32, 16, 128] | 1031 blocks, 32 layers, 32 kv_heads, block_size=16, head_dim=128

    # k_cache_new, v_cache_new = new keys/values (from prefill + previous decode steps)

    # block_table = maps each sequence ID to the physical blocks of KV cache it owns
    # block_table shape: [max_seqs_in_block_table, max_blocks_per_seq]
    # such as [128, 512] | For each sequence, maps up to 512 logical blocks to physical block ids

    # o = output tokens of the paged attention
    # o shape: [num_decoding_seqs, layer_hidden_dimension (num_q_heads*head_dim)]
    # such as [18, 4096] | Output tokens — probably flattened, where 4096 = 32 heads × 128 dims

    start_time_paged_attention = time.perf_counter()
    # mid_o = intermediate attention scores for each decoding sequence
    # mid_o shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    # such as [18, 32, 4, 128] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence, head_dim = 128
    mid_o = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)

    # mid_o_logexpsum = log-sum-exp of the intermediate attention scores for each decoding sequence
    # mid_o_logexpsum shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    # acc_buf = accumulated attention scores for each decoding sequence
    # acc_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    # such as [18, 32, 4, 128] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence, head_dim = 128
    acc_buf = torch.zeros((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)

    # sum_exp_buf = sum of the exponential of the attention scores for each decoding sequence
    # sum_exp_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
    sum_exp_buf = torch.zeros((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    # max_score_buf = maximum attention score for each decoding sequence
    # max_score_buf shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
    max_score_buf = torch.full((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), fill_value=-1e20, device=q.device, dtype=torch.float32)

    end_time_intermediate_outputs_init = time.perf_counter()

    def _launch(kc, vc, bt, new_kv_cache_used):
        # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
        # such as [18, 32, 4] | 18 decoding sequences, 32 attention heads, 4 blocks for each decoding sequence
        # Each thread in the grid is responsible for computing the attention output for:
        #   - a decoding sequence b
        #   - a query head h
        #   - a chunk of its history defined by my_seq_block_id
        grid = (infer_state.num_decoding_seqs, model_config.num_q_heads, infer_state.num_seq_blocks)
        if new_kv_cache_used:
            _fwd_paged_attention_phase1_new_kv_used[grid](
                acc_buf, sum_exp_buf, max_score_buf,
                q, k_cache, v_cache,
                block_table,
                infer_state.softmax_scale * 1.442695040888963,
                infer_state.decoding_seq_lens,
                infer_state.seq_ids[infer_state.num_prefill_seqs:],
                infer_state.num_seq_blocks,
                cur_layer,
                model_config.num_layers,
                model_config.num_q_heads,
                model_config.num_kv_heads,
                model_config.num_q_heads // model_config.num_kv_heads,
                engine_config.block_size,
                model_config.head_dim,
                infer_state.seq_block_size,
                engine_config.max_blocks_per_seq,
                num_warps = 1,
                num_stages = 4
            )
        else:
             _fwd_paged_attention_phase1[grid](
                mid_o, mid_o_logexpsum,
                q, k_cache, v_cache,
                block_table,
                infer_state.softmax_scale * 1.442695040888963,
                infer_state.decoding_seq_lens,
                infer_state.seq_ids[infer_state.num_prefill_seqs:],
                infer_state.num_seq_blocks,
                cur_layer,

                model_config.num_layers,
                model_config.num_q_heads,
                model_config.num_kv_heads,
                model_config.num_q_heads // model_config.num_kv_heads,
                engine_config.block_size,
                model_config.head_dim,
                infer_state.seq_block_size,
                engine_config.max_blocks_per_seq,
                num_warps = 1,
                num_stages = 4
            )

    new_kv_cache_used = False
    # 0-1) if there are no new kv caches, use the original kv cache
    num_new_groups = len(k_cache_new)
    if num_new_groups == 0:
        _launch(kc=k_cache, vc=v_cache, bt=block_table, new_kv_cache_used=new_kv_cache_used)
    else:
        # 0-2) if there are new kv caches, but all of them are not used, use the original kv cache
        num_blocks_org = infer_state.num_blocks_org
        mask0 = (block_table >= num_blocks_org)
        if not mask0.any():
            # print(f"~~~~ paged_attention for layer {cur_layer}, all new kv caches are not used ~~~~")
            _launch(kc=k_cache, vc=v_cache, bt=block_table, new_kv_cache_used=new_kv_cache_used)
        else:
            # print(f"~~~~ paged_attention for layer {cur_layer}, some new kv caches are used ~~~~~~")
            # print(f"~~~~ q shape: {q.shape}")
            # print(f"~~~~ k_cache shape: {k_cache.shape}")
            # print(f"~~~~ k_cache_new[0] shape: {k_cache_new[0].shape}")
            # print(f"~~~~ k_cache_new length: {len(k_cache_new)}")
            # print(f"~~~~ o shape: {o.shape}")
            # print(f"~~~~ block_table shape: {block_table.shape}, block_table: {block_table}")
            # print(f"infer_state.seq_ids: {infer_state.seq_ids}")
            # print(f"infer_state.num_decoding_seqs: {infer_state.num_decoding_seqs}")
            # print(f"infer_state.num_q_heads: {model_config.num_q_heads}")
            # print(f"infer_state.num_kv_heads: {model_config.num_kv_heads}")
            # print(f"infer_state.num_seq_blocks: {infer_state.num_seq_blocks}")
            # print(f"infer_state.seq_block_size: {infer_state.seq_block_size}")
            # print(f"infer_state.decoding_seq_lens: {infer_state.decoding_seq_lens}")
            # print(f"infer_state.max_blocks_per_seq: {engine_config.max_blocks_per_seq}")
            # print(f"infer_state.softmax_scale: {infer_state.softmax_scale}")
            # print(f"infer_state.prefill_seq_lens: {infer_state.prefill_seq_lens}")
            # print(f"infer_state.prefill_seq_start_locs: {infer_state.prefill_seq_start_locs}")
            # print(f"infer_state.num_prefill_seqs: {infer_state.num_prefill_seqs}")
            # print(f"infer_state.num_prefill_tokens: {infer_state.num_prefill_tokens}")
            # print(f"infer_state.max_prefill_len: {infer_state.max_prefill_len}")
            # print(f"~~~~ mask0 shape: {mask0.shape}, mask0: {mask0}")
            # print(f"\n~~~~ before block out ~~~~~~")
            # index_list = []
            # for index in range(block_table.shape[0]):
            #     if num_blocks_org <= max(block_table[index]):
            #         index_list.append(index)
            #         print(f"~~~index: {index} ~~~~")
            #         print(f"block_table[{index}].shape: {block_table[index].shape}, block_table[{index}] first 256: {block_table[index][:256]}")
            
            # 1) there are new kv caches, and some of them are used;
            # block out the blocks exceed the original kv cache size, and use the original kv caches. 
            new_kv_cache_used = True
            # Do not need to record event here, because all the kernels are launched in the same stream,
            # and the sequence of the kernels is deterministic.
            bt_org = block_table.clone()
            bt_org[mask0] = -1
            # print(f"\n~~~~ after block out ~~~~~~")
            # for index in index_list:
            #     print(f"~~~index: {index} ~~~~")
            #     print(f"bt_org[{index}].shape: {bt_org[index].shape}, bt_org[{index}] first 256: {bt_org[index][:256]}")
            # exit()
            _launch(kc=k_cache, vc=v_cache, bt=bt_org, new_kv_cache_used=new_kv_cache_used)
            # TODO: verify the correctness of the new KV cache being load and used...
            # 2) Run on the new kv caches, if some of them within a certain range are used.
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
                btj = btj - start
                btj[mask] = -1
                _launch(kc=k_cache_new[j], vc=v_cache_new[j], bt=btj, new_kv_cache_used=new_kv_cache_used)
            
    end_time_paged_attention_phase1 = time.perf_counter()

    # if there are new kv caches, we need to complete the calculation of the intermediate outputs: mid_o, mid_o_logexpsum
    # otherwise, we have already calculated the intermediate outputs in the paged attention phase1
    if new_kv_cache_used:
        # After the paged attention phase1, we have the intermediate buffers acc_buf, sum_exp_buf, max_score_buf
        # Complete the calculation of the intermediate outputs: mid_o, mid_o_logexpsum
        grid = (infer_state.num_decoding_seqs, model_config.num_q_heads, infer_state.num_seq_blocks)
        _paged_attention_mid_phase[grid](
            acc_buf,
            sum_exp_buf,
            max_score_buf,
            mid_o,
            mid_o_logexpsum,
            model_config.head_dim,
            infer_state.num_seq_blocks,
            num_warps=1,
            num_stages=2
        )
    
    end_time_paged_attention_phase_mid = time.perf_counter()

    grid = (infer_state.num_decoding_seqs, model_config.num_q_heads)
    _fwd_paged_attention_phase2[grid](
        mid_o, mid_o_logexpsum,
        o,
        infer_state.decoding_seq_lens,
        model_config.num_q_heads,
        model_config.head_dim,
        infer_state.num_seq_blocks,
        infer_state.seq_block_size,
    )

    end_time_paged_attention_phase2 = time.perf_counter()

    time_threshold = 0.5 # 500 ms
    if (end_time_paged_attention_phase2 - start_time_paged_attention) > time_threshold:
        print(f"\n[PagedAttention-Serving] PagedAttention serving time exceeded {time_threshold} s with {(end_time_paged_attention_phase2 - start_time_paged_attention):.2f} s.")
        print(f"[PagedAttention-Serving] Time to intermediate_outputs_init: {(end_time_intermediate_outputs_init - start_time_paged_attention)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Time to paged_attention_phase1:    {(end_time_paged_attention_phase1 - end_time_intermediate_outputs_init)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Time to paged_attention_phase_mid: {(end_time_paged_attention_phase_mid - end_time_paged_attention_phase1)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Time to paged_attention_phase2:    {(end_time_paged_attention_phase2 - end_time_paged_attention_phase_mid)*1000:.2f} ms.")
        print(f"[PagedAttention-Serving] Total time: {(end_time_paged_attention_phase2 - start_time_paged_attention)*1000:.2f} ms.")


def paged_attention_old_version(
    q: torch.Tensor,                    # [num_decoding_seqs, num_q_heads, head_dim]
    k_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache: torch.Tensor,              # [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    k_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache_new: torch.Tensor,          # List of [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    block_table: torch.Tensor,          # [max_seqs_in_block_table, max_blocks_per_seq]
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int,
    o: torch.Tensor                     # [num_decoding_seqs, num_q_heads, head_dim]
):
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_block_size % engine_config.block_size == 0
    assert o.is_contiguous()

    # q = current query tokens for each decoding sequence
    # k_cache, v_cache = previously cached keys/values (from prefill + earlier decode steps)
    # block_table = maps each sequence ID to the physical blocks of KV cache it owns

    # mid_o = intermediate attention scores for each decoding sequence
    # mid_o shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    mid_o = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)

    # mid_o_logexpsum = log-sum-exp of the intermediate attention scores for each decoding sequence
    # mid_o_logexpsum shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    grid = (infer_state.num_decoding_seqs, model_config.num_q_heads, infer_state.num_seq_blocks)
    _fwd_paged_attention_phase1[grid](
        mid_o, mid_o_logexpsum,
        q, k_cache, v_cache,
        block_table,

        # Here we multiply softmax_scale by log2(e) and use `exp2` instead of
        # `exp` because of two reasons:
        # 1. Up to 12 Jun 2024, all NVIDIA GPUs does not have a `exp` instruction
        #    in PTX. When calculating `exp`, they multiply the input by log2(e)
        #    and use `exp2` instead.
        # 2. Some optimizations are disabled while using `exp` in a loop, see
        #    https://github.com/triton-lang/triton/issues/2961
        infer_state.softmax_scale * 1.442695040888963,
        infer_state.decoding_seq_lens,
        infer_state.seq_ids[infer_state.num_prefill_seqs:],
        infer_state.num_seq_blocks,
        cur_layer,

        model_config.num_layers,
        model_config.num_q_heads,
        model_config.num_kv_heads,
        model_config.num_q_heads // model_config.num_kv_heads,
        engine_config.block_size,
        model_config.head_dim,
        infer_state.seq_block_size,
        engine_config.max_blocks_per_seq,
        num_warps = 1,
        num_stages = 4
    )

    grid = (infer_state.num_decoding_seqs, model_config.num_q_heads)
    _fwd_paged_attention_phase2[grid](
        mid_o, mid_o_logexpsum,
        o,
        infer_state.decoding_seq_lens,
        model_config.num_q_heads,
        model_config.head_dim,
        infer_state.num_seq_blocks,
        infer_state.seq_block_size,
    )

    # from swiftllm.utils import cdiv
    # for my_batch_id in range(infer_state.num_decoding_seqs):
    #     my_q = q[my_batch_id]   # [num_q_heads, head_dim]
    #     my_block_table = block_table[infer_state.seq_ids[infer_state.num_prefill_seqs+my_batch_id]]
    #     my_num_blocks = cdiv(infer_state.decoding_seq_lens[my_batch_id], engine_config.block_size)
    #     my_k_blocks = []
    #     my_v_blocks = []
    #     for block_id in range(my_num_blocks):
    #         block_index = my_block_table[block_id]
    #         my_k_blocks.append(k_cache[block_index][cur_layer])
    #         my_v_blocks.append(v_cache[block_index][cur_layer])
    #     my_k = torch.cat(my_k_blocks, dim=1)   # [num_kv_heads, *, head_dim]
    #     my_v = torch.cat(my_v_blocks, dim=1)   # [num_kv_heads, *, head_dim]
    #     my_k = my_k.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=0)   # [num_q_heads, *, head_dim]
    #     my_v = my_v.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=0)   # [num_q_heads, *, head_dim]
    #     my_q = my_q.reshape(model_config.num_q_heads, 1, model_config.head_dim)

    #     my_q = my_q.to(torch.float32)
    #     my_k = my_k.to(torch.float32)
    #     my_v = my_v.to(torch.float32)

    #     my_attn_score = torch.bmm(my_q, my_k.transpose(1, 2)).squeeze()   # [num_q_heads, *]
    #     my_attn_score = my_attn_score * infer_state.softmax_scale
    #     # print(my_v[0])
    #     # print(my_q[0])
    #     my_attn_score = torch.where(
    #         torch.arange(my_attn_score.shape[1], device=my_attn_score.device) < infer_state.decoding_seq_lens[my_batch_id],
    #         my_attn_score,
    #         torch.full_like(my_attn_score, float('-1e20'))
    #     )
    #     # print(my_attn_score)
    #     my_attn_score = torch.softmax(my_attn_score, dim=1)   # [num_q_heads, *]
    #     my_attn_score = my_attn_score.unsqueeze(1)   # [num_q_heads, 1, *]

    #     res = torch.bmm(my_attn_score, my_v).squeeze(1)   # [num_q_heads, head_dim]
    #     o[my_batch_id] = res.reshape(-1).to(torch.float16)
