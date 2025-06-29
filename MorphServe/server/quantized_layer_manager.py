import torch
import gc
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from awq.quantize.quantizer import real_quantize_model_weight
from awq.quantize.qmodule import WQLinear
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig
from swiftllm_c import (
    register_layer_memory_quant,
    register_layer_memory_tensor_map_quant,
)

class AWQLayerManager:
    def __init__(self, quantized_model_path, model_config, engine_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.pinned_memory = None
        self.layer_offsets = {}  # Start offset of each layer in pinned memory
        self.layer_sizes = {}    # Memory size of each layer
        self.tensor_maps_layer_addr = {}    # Maps from tensor name to offset for each layer
        self.tensor_maps_layer_id = {}    # Maps from tensor name to offset for each layer
        self.cpu_layers = {}     # Store reconstructed CPU layers
        self.active_layers = {}  # Layers currently on GPU
        self.layer_streams = {}  # CUDA streams for transfers
        self.feature_map = {}
        
        # Load model and create continuous memory layout
        self._load_model_and_extract_layers(quantized_model_path)

        # Extract features from the model LlamaDecoderLayer
        self._feature_extraction()
    
        # Prepare decoder layers in pinned memory
        self._organize_layers_in_continuous_memory()
        
        # Pre-compute CPU layers for quick access
        # self._precompute_cpu_layers()
        
    def _load_model_and_extract_layers(self, model_path):
        """Load AWQ quantized model and extract decoder layers"""
        print(f"[AWQLayerManager] Loading quantized model from {model_path}...")
        
        # Load model configuration
        config = AutoConfig.from_pretrained(
            self.engine_config.org_model_path, 
            trust_remote_code=True
        )
        config.use_cache = False
        
        # Create model with empty weights first
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            )
        
        # Initialize quantized weights structure
        real_quantize_model_weight(
            model, 
            w_bit=self.engine_config.quantized_q_config["w_bit"], 
            q_config=self.engine_config.quantized_q_config, 
            init_only=True
        )
        model.tie_weights()
        
        # Load actual weights
        # checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=False, assign=True)
        del checkpoint

        # print("\n", "[AWQLayerManager] model:")
        # print(model)

        # Store decoder layers temporarily
        self.temp_layers = []
        num_layers = len(model.model.layers)
        # print(f"[AWQLayerManager] Extracted {num_layers} decoder layers")
        # Copy layers to avoid references to original model
        for i in range(num_layers):
            self.temp_layers.append(model.model.layers[i])

        # Free original model to save memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def _feature_extraction(self):
        """
        Extract in_features and out_features from linear layers of the first LlamaDecoderLayer.
        """
        if not hasattr(self, 'temp_layers') or not self.temp_layers:
            print("[AWQLayerManager] No temporary layers available for feature extraction.")
            return
        
        # Get the first layer as the reference
        reference_layer = self.temp_layers[0]
        # print("[AWQLayerManager] Extracting linear layer features")
        
        # Initialize feature map for dimensions
        self.feature_map['dimensions'] = {}
        
        # Get dimensions from attention modules
        for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            module = getattr(reference_layer.self_attn, module_name)
            self.feature_map['dimensions'][f'self_attn.{module_name}'] = {
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        
        # Get dimensions from MLP modules
        for module_name in ['gate_proj', 'up_proj', 'down_proj']:
            module = getattr(reference_layer.mlp, module_name)
            self.feature_map['dimensions'][f'mlp.{module_name}'] = {
                'in_features': module.in_features,
                'out_features': module.out_features,
            }

        # print("[AWQLayerManager] Feature map:", self.feature_map)
        # print("[AWQLayerManager] Linear layer feature extraction complete")

    def _organize_layers_in_continuous_memory(self):
        """Pack all layers into one continuous pinned memory block"""
        # print(f"[AWQLayerManager] Organizing layers in continuous memory...")
        
        # Calculate total memory needed
        total_size = 0
        for i, layer in enumerate(self.temp_layers):
            # Calculate memory size for this layer
            layer_size = self._calculate_layer_size(layer)
            # Add padding for alignment (256-byte alignment)
            aligned_size = ((layer_size + 255) // 256) * 256
            self.layer_sizes[i] = aligned_size
            total_size += aligned_size
            # print(f"\n[AWQLayerManager] Layer {i}: addr: {id(layer)} (hex: {hex(id(layer))}), layer_size: {layer_size} Bytes, aligned_size: {aligned_size} Bytes")
            # register_layer_memory_quant(i, id(layer), aligned_size)
        # print(f"[AWQLayerManager] Total memory required: {total_size} Bytes")
        
        # Allocate continuous pinned memory
        self.pinned_memory = torch.empty(total_size, dtype=torch.uint8, device="cpu", pin_memory=True)

        # start_address = self.pinned_memory.data_ptr()
        # print("Start address:", start_address, "hex:", hex(start_address))

        # Copy layers to pinned memory
        offset = 0
        for i, layer in enumerate(tqdm(self.temp_layers, desc="Copying layers to pinned memory")):
            self.layer_offsets[i] = offset
            self._copy_layer_to_pinned_memory(layer, i, offset)
            offset += self.layer_sizes[i]
            # Create CUDA stream for this layer
            self.layer_streams[i] = torch.cuda.Stream()
        
        # Free temporary layers
        self.temp_layers = None
        gc.collect()
        torch.cuda.empty_cache()


    def _calculate_layer_size(self, layer):
        """Calculate total memory size needed for a layer"""
        total_bytes = 0
        tensor_map = {}
        # Get parameters
        for name, param in layer.named_parameters():
            # print(f"\n[AWQLayerManager] param: {name}, shape: {param.shape}, dtype: {param.dtype}, size: {param.numel() * param.element_size()} Bytes")
            tensor_bytes = param.numel() * param.element_size()
            tensor_map[name] = {
                'shape': param.shape,
                'dtype': param.dtype,
                'size': tensor_bytes,
                'is_param': True
            }
            total_bytes += tensor_bytes
        # Get buffers
        for name, buffer in layer.named_buffers():
            # print(f"\n[AWQLayerManager] buffer: {name}, shape: {buffer.shape}, dtype: {buffer.dtype}, size: {buffer.numel() * buffer.element_size()} Bytes")
            if name == "self_attn.rotary_emb.inv_freq":
                # print(f"\n[AWQLayerManager] buffer: {name}, shape: {buffer.shape}, dtype: {buffer.dtype}, size: {buffer.numel() * buffer.element_size()} Bytes")
                continue
            tensor_bytes = buffer.numel() * buffer.element_size()
            tensor_map[name] = {
                'shape': buffer.shape,
                'dtype': buffer.dtype,
                'size': tensor_bytes,
                'is_param': False
            }
            total_bytes += tensor_bytes
        # Store tensor map for this layer
        self.tensor_maps_layer_addr[id(layer)] = tensor_map
        return total_bytes
    

    def _copy_layer_to_pinned_memory(self, layer, layer_id, offset):
        """Copy all tensors of a layer to pinned memory in a continuous block"""
        # Create tensor map for this specific layer instance
        self.tensor_maps_layer_id[layer_id] = {}
        tensor_map = self.tensor_maps_layer_addr[id(layer)]
        tensor_map_list = []
        # Copy parameters and track offsets
        current_offset = offset
        # Process parameters
        for name, param in layer.named_parameters():
            info = tensor_map[name]
            size = info['size']
            # print("\n[AWQLayerManager] name:", name)
            # print("[AWQLayerManager] param:", param)
            # print("[AWQLayerManager] info:", info)
            # print("[AWQLayerManager] size:", size)
            # Create view into pinned memory
            tensor_bytes_view = self.pinned_memory[current_offset:current_offset + size]
            # Convert tensor to bytes and copy
            tensor_bytes = param.detach().cpu().numpy().view(np.uint8).reshape(-1)
            tensor_bytes_view.copy_(torch.from_numpy(tensor_bytes))
            # print("tensor_bytes:", tensor_bytes)
            # print("tensor_bytes_view:", tensor_bytes_view)
            # Store metadata with absolute offset
            self.tensor_maps_layer_id[layer_id][name] = {
                'offset': current_offset - offset,  # Relative to layer start
                'shape': info['shape'],
                'dtype': info['dtype'],
                'size': size,
                'is_param': True
            }
            # Update offset
            current_offset += size
            tensor_map_list.append({name: self.tensor_maps_layer_id[layer_id][name]})
        
        # Process buffers
        for name, buffer in layer.named_buffers():
            if name == "self_attn.rotary_emb.inv_freq":
                continue
            info = tensor_map[name]
            size = info['size']
            
            # print("\n[AWQLayerManager] name:", name)
            # print("[AWQLayerManager] buffer:", buffer)
            # print("[AWQLayerManager] info:", info)
            # print("[AWQLayerManager] size:", size)

            # Create view into pinned memory
            tensor_bytes_view = self.pinned_memory[current_offset:current_offset + size]
            
            # Convert tensor to bytes and copy
            tensor_bytes = buffer.detach().cpu().numpy().view(np.uint8).reshape(-1)
            tensor_bytes_view.copy_(torch.from_numpy(tensor_bytes))
            # print("tensor_bytes:", tensor_bytes)
            # print("tensor_bytes_view:", tensor_bytes_view)

            # Store metadata with absolute offset
            self.tensor_maps_layer_id[layer_id][name] = {
                'offset': current_offset - offset,  # Relative to layer start
                'shape': info['shape'],
                'dtype': info['dtype'],
                'size': size,
                'is_param': False
            }
            
            # Update offset
            current_offset += size
            tensor_map_list.append({name: self.tensor_maps_layer_id[layer_id][name]})

        # print("\n[AWQLayerManager] layer_id:", layer_id)
        register_layer_memory_quant(layer_id, self.pinned_memory.data_ptr() + offset, current_offset - offset)
        # print("tensor_maps:", self.tensor_maps_layer_id[layer_id])
        register_layer_memory_tensor_map_quant(layer_id, tensor_map_list)

    
    def transfer_layer_to_gpu(self, layer_id):
        """Transfer a layer from CPU to GPU efficiently"""
        if layer_id not in self.cpu_layers:
            raise ValueError(f"Layer {layer_id} not found in CPU layers")
        # Get CUDA stream for transfer
        stream = self.layer_streams[layer_id]
        # Transfer using dedicated stream
        with torch.cuda.stream(stream):
            # Move layer to GPU with single operation
            gpu_layer = self.cpu_layers[layer_id].to("cuda")
            # Store in active layers
            self.active_layers[layer_id] = gpu_layer
        # Wait for transfer to complete
        torch.cuda.current_stream().wait_stream(stream)
        return self.active_layers[layer_id]
    
    def free_layer_memory(self, layer_id):
        """Free GPU memory for a specific layer"""
        if layer_id not in self.active_layers:
            return False
        # Get layer
        layer = self.active_layers[layer_id]
        # Clear all parameters and buffers
        for name, param in list(layer.named_parameters()):
            param.data = None
        for name, buffer in list(layer.named_buffers()):
            buffer.data = None
        # Remove from active layers
        del self.active_layers[layer_id]
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        return True
   

    def check_continuity_and_memory(self, layer):
        total_layer_size = 0
        tensors = {
            "rotary_emb_inv_freq": layer.self_attn.rotary_emb.inv_freq,
            "q_proj_qweight": layer.self_attn.q_proj.qweight,
            "q_proj_scales": layer.self_attn.q_proj.scales,
            "q_proj_scaled_zeros": layer.self_attn.q_proj.scaled_zeros,
            "k_proj_qweight": layer.self_attn.k_proj.qweight,
            "k_proj_scales": layer.self_attn.k_proj.scales,
            "k_proj_scaled_zeros": layer.self_attn.k_proj.scaled_zeros,
            "v_proj_qweight": layer.self_attn.v_proj.qweight,
            "v_proj_scales": layer.self_attn.v_proj.scales,
            "v_proj_scaled_zeros": layer.self_attn.v_proj.scaled_zeros,
            "o_proj_qweight": layer.self_attn.o_proj.qweight,
            "o_proj_scales": layer.self_attn.o_proj.scales,
            "o_proj_scaled_zeros": layer.self_attn.o_proj.scaled_zeros,
            "gate_proj_qweight": layer.mlp.gate_proj.qweight,
            "gate_proj_scales": layer.mlp.gate_proj.scales,
            "gate_proj_scaled_zeros": layer.mlp.gate_proj.scaled_zeros,
            "up_proj_qweight": layer.mlp.up_proj.qweight,
            "up_proj_scales": layer.mlp.up_proj.scales,
            "up_proj_scaled_zeros": layer.mlp.up_proj.scaled_zeros,
            "down_proj_qweight": layer.mlp.down_proj.qweight,
            "down_proj_scales": layer.mlp.down_proj.scales,
            "down_proj_scaled_zeros": layer.mlp.down_proj.scaled_zeros,
            "attn_norm_weight": layer.input_layernorm.weight,
            "ffn_norm_weight": layer.post_attention_layernorm.weight,
        }
        tensor_info = []
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                start_address = tensor.data_ptr()
                end_address = start_address + tensor.numel() * tensor.element_size()
                total_layer_size += tensor.numel() * tensor.element_size()
                tensor_info.append({
                    "name": name,
                    "start_address": start_address,
                    "end_address": end_address,
                    "is_contiguous": tensor.is_contiguous(),
                    "total_size": tensor.numel() * tensor.element_size()
                })
        # Sort by starting address
        tensor_info.sort(key=lambda x: x["start_address"])
        # Display the results
        print(f"\n[AWQLayerManager] ~~~~~~~~~~~~~~ Tensor device: {tensors['attn_norm_weight'].device} ~~~~~~~~~~~~~~")
        for i, info in enumerate(tensor_info):
            print(f"\n🔍 Tensor: {info['name']}")
            print(f"  - Start address: {hex(info['start_address'])}")
            print(f"  - End address: {hex(info['end_address'])}")
            print(f"  - Is contiguous? {'✅ Yes' if info['is_contiguous'] else '❌ No'}")
            print(f"  - Total size (in bytes): {info['total_size']}")
            # Compare with the previous tensor's end address
            if i > 0:
                gap = info['start_address'] - tensor_info[i-1]['end_address']
                print(f"  - 🟠 Gap from previous tensor: {gap} bytes ({'Continuous ✅' if gap == 0 else 'Non-continuous ❌'})")
        print(f"  - Total layer size: {total_layer_size} bytes")
        print(f"  - Total layer size (in MB): {total_layer_size / (1024 * 1024):.2f} MB")
        print(f"  - Total layer size (in GB): {total_layer_size / (1024 * 1024 * 1024):.2f} GB")

