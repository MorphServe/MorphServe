import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        
        # --- New Code: Compute and print logits for all tokens ---
        # for PPL evaluation, we need to return all_logits_with_rms with each request 
        # print("~"*100)
        # print("input_embds.shape:", input_embds.shape)
        # rmsnorm_inplace(
        #     input_embds,
        #     self.weights.final_norm,
        #     self.model_config.rms_norm_eps
        # )
        # all_logits_with_rms = linear(input_embds, self.weights.lm_head)  # [total_token_num, vocab_size]
        # print("~"*100)
        # print("all_logits_with_rms.shape:", all_logits_with_rms.shape)
        # print("all_logits_with_rms.numel():", all_logits_with_rms.numel())
        # print("all_logits_with_rms:", all_logits_with_rms)

        # Assume all_logits_with_rms is your tensor with shape [x, y]
        # chunk_size = 2048
        # This will split the tensor along the first dimension into chunks of size 2048.
        # The last chunk will have a size less than 2048 if x % 2048 != 0.
        # chunks = list(torch.split(all_logits_with_rms, chunk_size, dim=0))
        # You can then iterate over the chunks:
        # for i, chunk in enumerate(chunks):
        #     print(f"Chunk {i} shape: {chunk.shape}")        
        # print("~"*100)
        # print("len(all_logits_with_rms_chunks):", len(chunks))
        # print("all_logits_with_rms_chunks[0].shape:", chunks[0].shape, "all_logits_with_rms_chunks[0]:", chunks[0])
        # for PPL evaluation, we need to return all_logits_with_rms
        # return chunks
        

        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)

        return output_tokens
    