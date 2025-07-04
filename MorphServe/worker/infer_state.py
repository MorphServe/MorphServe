import dataclasses
import torch

@dataclasses.dataclass
class LlamaInferState:
    batch_size: int
    num_tokens: int

    seq_ids: torch.Tensor   # [batch_size]
    softmax_scale: float    # Equal to 1/sqrt(head_dim)

    num_prefill_seqs: int
    num_prefill_tokens: int
    prefill_seq_start_locs: torch.Tensor # [batch_size]
    prefill_seq_start_locs_with_end: torch.Tensor # [batch_size+1], = prefill_seq_start_locs + [num_prefill_tokens]
    prefill_seq_lens: torch.Tensor # [batch_size]
    max_prefill_len: int

    num_decoding_seqs: int
    decoding_seq_lens: torch.Tensor # [batch_size]
    max_decoding_len: int

    seq_block_size: int
    num_seq_blocks: int

    num_blocks_org: int # The number of blocks in the original model, used to calculate the mask for the block table in the quant version of store_kvcache()
    kv_cache_new_block_size: int # The number of blocks in the new kv cache, used to calculate the mask for the block table in the quant version of store_kvcache()
    org_layer_param_size: int # The size of the original layer parameters. 

    position_cos: torch.Tensor	# [num_tokens, hidden_size]
    position_sin: torch.Tensor	# [num_tokens, hidden_size]

    ignore_kvcache: bool    # Skip storing the key/value cache, useful when profiling the number of kv blocks
