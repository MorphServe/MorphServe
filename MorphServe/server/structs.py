import asyncio
import dataclasses
import time
from typing import List
import pandas as pd
import torch

@dataclasses.dataclass
class StepOutput:
    """
    The output of one decoding step
    """
    token_id: int
    request: "Request"

@dataclasses.dataclass
class RequestMetrics:
    """
    Tracks performance metrics for each request
    """
    arrival_time: float = 0.0      # Time when the request arrives
    tokenized_time: float = 0.0    # Time when tokenization is complete
    queued_time: float = 0.0       # Time when req move from waiting to running
    first_token_time: float = 0.0  # Time when the first token is generated
    completion_time: float = 0.0   # Time when the request is fully completed

@dataclasses.dataclass
class ServerMetrics:
    """
    Tracks server performance metrics over time
    """
    timestamps: List[float] = dataclasses.field(default_factory=list)
    num_requests_waiting: List[int] = dataclasses.field(default_factory=list)
    num_requests_running: List[int] = dataclasses.field(default_factory=list)
    num_requests_swapping: List[int] = dataclasses.field(default_factory=list)
    num_preemptions: List[int] = dataclasses.field(default_factory=list)
    num_layer_quantized: List[int] = dataclasses.field(default_factory=list)
    gpu_total_blocks: List[int] = dataclasses.field(default_factory=list)
    gpu_kv_cache_usage: List[float] = dataclasses.field(default_factory=list)
    cpu_kv_cache_usage: List[float] = dataclasses.field(default_factory=list)
    prefilling_throughput: List[float] = dataclasses.field(default_factory=list)
    decoding_throughput: List[float] = dataclasses.field(default_factory=list)
    tokens_per_second: List[float] = dataclasses.field(default_factory=list)
    time_interval: float = 1.0
    prefill_tokens: int = 0
    prefill_time: float = 0
    decode_tokens: int = 0
    decode_time: float = 0
    preemption_count: int = 0

    def record_metrics(self, 
                      num_requests_waiting: int,
                      num_requests_running: int,
                      num_requests_swapping: int,
                      num_preemptions: int,
                      num_layer_quantized: int,
                      gpu_total_blocks: int,
                      gpu_kv_cache_usage: float,
                      cpu_kv_cache_usage: float,
                      prefilling_throughput: float,
                      decoding_throughput: float,
                      tokens_per_second: float):
        """Record a new set of metrics"""
        self.timestamps.append(time.perf_counter())
        self.num_requests_waiting.append(num_requests_waiting)
        self.num_requests_running.append(num_requests_running)
        self.num_requests_swapping.append(num_requests_swapping)
        self.num_preemptions.append(num_preemptions)
        self.num_layer_quantized.append(num_layer_quantized)
        self.gpu_total_blocks.append(gpu_total_blocks)
        self.gpu_kv_cache_usage.append(gpu_kv_cache_usage)
        self.cpu_kv_cache_usage.append(cpu_kv_cache_usage)
        self.prefilling_throughput.append(prefilling_throughput)
        self.decoding_throughput.append(decoding_throughput)
        self.tokens_per_second.append(tokens_per_second)

    def save_to_csv(self, filepath: str):
        """Save metrics data to a CSV file"""
        
        # Enhanced helper function to convert tensors to numpy arrays or native types
        def to_numpy(value):
            if isinstance(value, torch.Tensor):  # Check for all tensor types (CPU or CUDA)
                return value.cpu().numpy() if value.is_cuda else value.numpy()
            return value  # Return non-tensor values directly

        # Create a dictionary of all metrics, converting tensors to numpy arrays
        data = {
            'timestamp': [to_numpy(v) for v in self.timestamps],
            'num_requests_waiting': [to_numpy(v) for v in self.num_requests_waiting],
            'num_requests_running': [to_numpy(v) for v in self.num_requests_running],
            'num_requests_swapping': [to_numpy(v) for v in self.num_requests_swapping],
            'num_preemptions': [to_numpy(v) for v in self.num_preemptions],
            'num_layer_quantized': [to_numpy(v) for v in self.num_layer_quantized],
            'gpu_total_blocks': [to_numpy(v) for v in self.gpu_total_blocks],
            'gpu_kv_cache_usage': [to_numpy(v) for v in self.gpu_kv_cache_usage],
            'cpu_kv_cache_usage': [to_numpy(v) for v in self.cpu_kv_cache_usage],
            'prefilling_throughput': [to_numpy(v) for v in self.prefilling_throughput],
            'decoding_throughput': [to_numpy(v) for v in self.decoding_throughput],
            'tokens_per_second': [to_numpy(v) for v in self.tokens_per_second]
        }

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        print(f"\nServer metrics saved successfully to {filepath}")
        print(df.describe())
        # print(df.dtypes)
        # print(df.head(15))


class RawRequest:
    """
    A request issued by user
    """
    prompt: str
    output_len: int
    input_ids: list[int]
    tokenized: bool

    def __init__(self, prompt=None, output_len=None, input_ids=[]):
        assert (prompt is not None) or (input_ids)
        assert (output_len is not None)
        self.prompt = prompt
        self.output_len = output_len
        self.input_ids = input_ids
        self.tokenized = False
        if input_ids:
            self.tokenized = True

class Request:
    """
    A (queuing, processing, or finished) request in the system
    """   

    prompt_token_ids: list[int]     # Prompt token ids, generated by the tokenizer upon request arrival
    prompt_len: int     # len(prompt_token_ids)
    output_len: int     # Output length
    num_layers_quantized: int     # Number of layers quantized

    output_q: asyncio.Queue[StepOutput] # Queue initialized when the raw request enters the
                                        # engine, and to be set upon a new token being generated
                                        # Mainly for streaming the output back to the user
    finished_event: asyncio.Event       # Event to be set when the request is finished
                                        # Mainly for the non-streaming case

    request_id: int     # Request ID, within range [0, max_seqs_in_block_table).
                        # Generated before being prefilled, and used as the index
                        # into the block table
    output_token_ids: list[int]     # Output token ids

    req_metrics: RequestMetrics    # New field for tracking performance metrics

    def __init__(self, raw_request: RawRequest):
        # A request is __init__-ed when entering `untokenized_raw_requests`, and
        # its `prompt_token_ids` and `prompt_len` will be set upon tokenization
        self.prompt_token_ids = raw_request.input_ids
        self.prompt_len = len(self.prompt_token_ids)
        self.num_layers_quantized = 0
        self.output_len = raw_request.output_len
        self.output_q = asyncio.Queue()
        self.finished_event = asyncio.Event()
        self.request_id = -1
        self.output_token_ids = []
        self.req_metrics = RequestMetrics(arrival_time=time.perf_counter())  # Initialize arrival time

    def is_finished(self) -> bool:
        return len(self.output_token_ids) == self.output_len
    
    def get_cur_output_len(self) -> int:
        return len(self.output_token_ids)

    def is_prefill_stage(self) -> bool:
        return not self.output_token_ids
    