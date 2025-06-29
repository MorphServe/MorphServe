import json
import os
import dataclasses
import torch
import safetensors
import numpy as np

from swiftllm.model_config import LlamaModelConfig

@dataclasses.dataclass
class RegisteredWeightItem:
    attr_name: str
    key: str
    shape: tuple
    dtype: torch.dtype

class WeightBase:
    """
    The base class of all weight classes (i.e. LlamaTransformerLayerWeight or LlamaWeight)

    During weight initialization, each concrete weight class should first register
    all weight items. Each weight item has its own attribute name, key, shape, and dtype.

    During weight loading, RegisterWeightItem will be passed to the weight getter
    function, which should return the corresponding weight value (real/dummy).
    """

    def __init__(self):
        self.registered_weights = []

    def register_weight(self, item: RegisteredWeightItem):
        self.registered_weights.append(item)

    def _post_process_after_load(self, getter: callable):
        """
        This function is called after loading weights (real/dummy).
        Defined in each concrete weight class, called by load_weights().
        """
        raise NotImplementedError()
    
    def load_weights(self, getter: callable):
        """
        Load weights
        """
        for item in self.registered_weights:
            # print(f"[Weight] loading {item.key}")
            weight_value = getter(item)
            assert weight_value is not None, f"getter() returned None for {item.key} ({item})"
            assert isinstance(weight_value, torch.Tensor), f"Weight {item.key} is not a tensor"
            assert weight_value.shape == item.shape, f"Shape of weight {item.key} does not match"
            assert weight_value.device.type in ["cuda", "cpu"], f"Weight {item.key} is not on GPU or CPU"
            setattr(self, item.attr_name, weight_value.to(item.dtype))
        self._post_process_after_load(getter)


class LlamaTransformerLayerWeight(WeightBase):
    """
    Class stores the weights of one transformer layer (transformer block) in Llama model.
    """

    def __init__(
        self,
        layer_id: int,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "attn_norm",
            f"model.layers.{self.layer_id}.input_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "q_proj",
            f"model.layers.{self.layer_id}.self_attn.q_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "k_proj",
            f"model.layers.{self.layer_id}.self_attn.k_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "v_proj",
            f"model.layers.{self.layer_id}.self_attn.v_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "o_proj",
            f"model.layers.{self.layer_id}.self_attn.o_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))

        self.register_weight(RegisteredWeightItem(
            "ffn_norm",
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "up_proj",
            f"model.layers.{self.layer_id}.mlp.up_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "gate_proj",
            f"model.layers.{self.layer_id}.mlp.gate_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "down_proj",
            f"model.layers.{self.layer_id}.mlp.down_proj.weight",
            (self.model_config.hidden_size, self.model_config.ffn_inter_dim),
            self.dtype
        ))

    def _post_process_after_load(self, getter: callable):
        # pylint: disable=no-member
        # pylint: disable=no-member
        # self.qkv_proj = torch.cat((self.q_proj, self.k_proj, self.v_proj), dim=0).contiguous()
        # del self.q_proj, self.k_proj, self.v_proj
        # self.up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).contiguous()
        # del self.up_proj, self.gate_proj

        # Since up_proj and gate_proj are stored back-to-back in our continuous buffer,
        # we can create a view combining them instead of concatenating
        assert self.up_proj.stride() == self.gate_proj.stride(), "Tensors must have same stride for continuous view"
        assert self.up_proj.data_ptr() + self.up_proj.nelement() * self.up_proj.element_size() == self.gate_proj.data_ptr(), "Tensors must be contiguous in memory"
        
        # Create a view of the combined tensors
        combined_shape = (self.up_proj.size(0) + self.gate_proj.size(0), self.up_proj.size(1))
        self.up_gate_proj = torch.as_strided(self.up_proj, 
                                           size=combined_shape,
                                           stride=self.up_proj.stride())
        
        # Delete individual tensors but keep the memory (since up_gate_proj is a view)
        del self.up_proj, self.gate_proj


class LlamaWeight(WeightBase):
    def __init__(
        self,
        model_config: LlamaModelConfig,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "wte",
            "model.embed_tokens.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "lm_head",
            "lm_head.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "final_norm",
            "model.norm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))

        self.layers: list[LlamaTransformerLayerWeight] = []
        # Create all layers
        for i in range(self.model_config.num_layers):
            layer = LlamaTransformerLayerWeight(i, self.model_config, self.dtype)
            self.layers.append(layer)

    def _post_process_after_load(self, getter: callable):
        for layer in self.layers:
            layer.load_weights(getter)


def load_weights(
    model_config: LlamaModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False,
    load_to_cpu: bool = False
) -> LlamaWeight:
    """
    Load weights from a given path using pinned memory for efficient CPU-to-GPU transfer
    Args:
        model_config: Model configuration
        dtype: Data type for weights
        model_path: Path to model weights
        use_dummy: Whether to use dummy weights
        load_to_cpu: If True, keep weights in CPU memory, otherwise transfer to GPU
    """
    if use_dummy:
        def weight_getter_dummy(item: RegisteredWeightItem):
            device = "cpu" if load_to_cpu else "cuda"
            return torch.empty(item.shape, dtype=item.dtype, device=device).uniform_(-0.001, 0.001)
        getter = weight_getter_dummy
    else:
        # Instantiate LlamaWeight instead of WeightBase
        temp_weight = LlamaWeight(model_config, dtype)  # Temporary to extract weight metadata
        # print(f"[Weight] temp_weight.registered_weights: {temp_weight.registered_weights}")
        # print(f"[Weight] temp_weight.layers: {temp_weight.layers}")
        # print(f"[Weight] temp_weight.layers[0].registered_weights: {temp_weight.layers[0].registered_weights}")
        
        # Initialize tracking variables
        current_offset = 0
        total_elements = 0
        weight_offsets = {}  # Store offsets for each weight in the continuous buffer

        # Calculate total elements for top-level registered weights
        for item in temp_weight.registered_weights:
            num_elements = np.prod(item.shape)
            weight_offsets[item.key] = (current_offset, num_elements, item.shape)
            current_offset += num_elements
            total_elements += num_elements

        # Calculate total elements for each transformer layer
        for layer in temp_weight.layers:
            for item in layer.registered_weights:
                num_elements = np.prod(item.shape)
                weight_offsets[item.key] = (current_offset, num_elements, item.shape)
                current_offset += num_elements
                total_elements += num_elements

        # Calculate memory buffer size
        element_size = torch.finfo(dtype).bits // 8
        total_memory_gb = total_elements * element_size / 1024**3

        print(f"[Weight] Allocating {total_memory_gb:.2f} GB pinned memory buffer")

        # Allocate pinned memory buffer
        pinned_buffer = torch.empty(total_elements, dtype=dtype, pin_memory=True)
        
        if not load_to_cpu:
            print(f"[Weight] Pre-allocating GPU buffer")
            gpu_buffer = torch.empty(total_elements, dtype=dtype, device="cuda")
        
        # print weight_offsets here for debugging
        # print(f"[Weight] weight_offsets: {weight_offsets}")
        
        safetensor_files = [name for name in os.listdir(model_path) if name.endswith(".safetensors")]
        if len(safetensor_files) > 0:
            # Use Safetensors
            safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(safetensor_index_path):
                # The weight is stored in multiple files
                f = open(safetensor_index_path, "r", encoding="utf-8")
                safetensor_index = json.load(f)["weight_map"]
                safetensor_filename = None
            else:
                # The weight is stored in a single file
                assert len(safetensor_files) == 1, "model.safetensors.index.json not found, but there are multiple .safetensors files"
                safetensor_index = None
                safetensor_filename = safetensor_files[0]

            def weight_getter_real(item: RegisteredWeightItem):
                file_name = safetensor_index[item.key] if safetensor_index is not None else safetensor_filename
                file_path = os.path.join(model_path, file_name)
                offset, num_elements, shape = weight_offsets[item.key]
                
                # Load directly into the pinned buffer at the correct offset
                with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(item.key)
                    pinned_buffer[offset:offset + num_elements].copy_(tensor.view(-1))
                
                if not load_to_cpu:
                    gpu_buffer[offset:offset + num_elements].copy_(pinned_buffer[offset:offset + num_elements], non_blocking=True)
                    return gpu_buffer[offset:offset + num_elements].view(shape)
                else:
                    return pinned_buffer[offset:offset + num_elements].view(shape)
            
            getter = weight_getter_real
        else:
            # Use PyTorch
            pytorch_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            if os.path.exists(pytorch_index_path):
                # The weight is stored in multiple files
                f = open(pytorch_index_path, "r", encoding="utf-8")
                pytorch_index = json.load(f)["weight_map"]
                pytorch_filename = None
            else:
                # The weight is stored in a single file
                pytorch_index = None
                pytorch_filename = "pytorch_model.bin"
            
            # For PyTorch files, since "opening" it is slow (due to deserialization),
            # we open it only once and then store the opened files in a dictionary.
            # We add `mmap=True` to avoid loading the entire file into memory.
            opened_files = {}
            def weight_getter_real(item: RegisteredWeightItem):
                file_name = pytorch_index[item.key] if pytorch_index is not None else pytorch_filename
                file_path = os.path.join(model_path, file_name)
                offset, num_elements, shape = weight_offsets[item.key]
                
                if file_path not in opened_files:
                    opened_files[file_path] = torch.load(file_path, map_location="cpu", mmap=True, weights_only=True)
                
                file = opened_files[file_path]
                # Copy the weight into the pinned buffer at the correct offset
                pinned_buffer[offset:offset + num_elements].copy_(file[item.key].view(-1))
                
                if not load_to_cpu:
                    # Copy from pinned memory to GPU buffer
                    gpu_buffer[offset:offset + num_elements].copy_(pinned_buffer[offset:offset + num_elements], non_blocking=True)
                    return gpu_buffer[offset:offset + num_elements].view(shape)
                else:
                    return pinned_buffer[offset:offset + num_elements].view(shape)
            
            getter = weight_getter_real

    # Base class: BaseWeight
    # Derived class: LlamaWeight
    # Derived class: LlamaTransformerLayerWeight

    # First, create the derived class object (LlamaWeight), 
    # which will call the register_weight() method of the base class (BaseWeight)
    # and register all the weights (3 registered weights) and self.layers with 32 layers;

    # Then, call the load_weights() method of the derived class (LlamaWeight)
    # The load_weights() method will call the weight_getter() within the above weight_getter_real function 
    # and load 3 registered weights; and perform the post-processing after loading weights; LlamaWeight._post_process_after_load

    # Next, The 32 LlamaTransformerLayerWeight objects will be loaded; 
    # each LlamaTransformerLayerWeight object will call the load_weights() method of the base class (WeightBase)
    # and perform the post-processing after loading weights of the LlamaTransformerLayerWeight._post_process_after_load.

    weight = LlamaWeight(model_config, dtype)
    weight.load_weights(getter)
    # print(f"[Weight] weight.registered_weights: {weight.registered_weights}")
    # print(f"[Weight] weight.layers: {weight.layers}")
    # print(f"[Weight] weight.layers[0].registered_weights: {weight.layers[0].registered_weights}")
    
    if not load_to_cpu:
        # Make sure all GPU copies are complete
        torch.cuda.synchronize()
        # Free pinned memory since we don't need it anymore
        # This will help release system resources
        print("[Weight] Freeing pinned CPU memory...")
        del pinned_buffer
        torch.cuda.empty_cache()  # Help ensure memory is freed
    
    print(f"[Weight] model loaded successfully.")

    return weight
