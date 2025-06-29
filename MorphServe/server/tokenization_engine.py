import ray
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig

@ray.remote
class TokenizationEngine:
    '''
    Why Use Ray Here?
        Tokenization is typically CPU-bound, so using Ray enables efficient parallel execution on 
        multiple CPU cores, boosting throughput in scenarios with many concurrent requests.

    Why use ray.remote?
        The .remote() method runs the TokenizationEngine in a separate Ray worker process, 
        allowing concurrent tokenization requests across CPU cores.
    
    How Does Ray Actor Model Improve Tokenization?
        In a standard synchronous workflow, tokenizing thousands of prompts may create bottlenecks. 
        Ray actor model addresses this by:
            Running the TokenizationEngine as an independent process to isolate workloads. 
            Leveraging concurrent execution by spawning multiple tokenization workers. 
            Dynamically scaling tokenization tasks across multiple CPU cores for improved throughput.
    '''
    def __init__(self, engine_config: EngineConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(engine_config.org_model_path)

    def batched_tokenize(self, prompts: list[str]) -> list[list[int]]:
        prompt_token_ids = self.tokenizer(prompts, return_attention_mask=False)['input_ids']
        return prompt_token_ids
