import torch
import time
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from awq.quantize.qmodule import WQLinear
import torch.nn as nn
from swiftllm.worker.model import BlockManager
import gc
from swiftllm_c import (
    register_layer_memory_org_gpu,
    register_layer_memory_org_cpu,
    register_layer_memory_quant,
    register_layer_memory_tensor_map_quant,
    register_layer_memory_tensor_map_org,
    replace_layer_org2quant,
    replace_layer_quant2org,
    acquire_new_kvcache
)
from accelerate import init_empty_weights
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.layers.transformer_layer import LlamaTransformerLayer
from swiftllm.worker.layers.transformer_layer import AWQTransformerLayer


@torch.inference_mode()
def initialize_memory_tracking(model, device_on_gpu=True):
    """Register all layer memory ranges during initialization"""
    for layer_id in range(len(model.transformer_layers)):
        layer = model.transformer_layers[layer_id]
        
        # Find memory address range
        tensors = []
        tensor_map_list = []
        tensor_info = {}
        element_size = 0
        # First pass: collect all tensors and their info
        for attr_name in dir(layer.weight):
            attr = getattr(layer.weight, attr_name)
            if isinstance(attr, torch.Tensor):
                if (device_on_gpu and attr.device.type == 'cuda') or (not device_on_gpu and attr.device.type == 'cpu'):
                    # Store tensor metadata
                    size_bytes = attr.numel() * attr.element_size()
                    tensors.append((attr_name, attr.data_ptr(), size_bytes))
                    element_size = attr.element_size()
                    # Store detailed tensor info for mapping
                    tensor_info[attr_name] = {
                        'shape': attr.shape,
                        'dtype': attr.dtype,
                        'size': size_bytes,
                        'is_param': True  # Assuming all are parameters in the weight object
                    }

        if not tensors:
            print(f"[Memory] Warning: No tensors found for layer {layer_id}")
            continue

        # Sort tensors by memory address to find contiguous range
        tensors.sort(key=lambda x: x[1])
        
        # Calculate range
        start_addr = tensors[0][1]
        end_addr = tensors[-1][1] + tensors[-1][2]
        total_size = end_addr - start_addr
        model.org_layer_param_size = int(total_size / element_size)

        # Calculate relative offsets for each tensor
        for attr_name, data_ptr, size in tensors:
            offset = data_ptr - start_addr
            
            # Update tensor info with offset relative to layer start
            tensor_info[attr_name]['offset'] = offset
            
            # Add to tensor map list in the format expected by C++
            tensor_map_list.append({attr_name: tensor_info[attr_name]})

        # Pass the memory address as an integer (matching the C++ signature)
        if device_on_gpu:
            # Register the memory address for the original layer
            register_layer_memory_org_gpu(layer_id, start_addr, total_size)
        else:
            # Register the memory address for the original layer
            register_layer_memory_org_cpu(layer_id, start_addr, total_size)
            # Register the tensor map for the original layer
            register_layer_memory_tensor_map_org(layer_id, tensor_map_list)

        # print(f"[Memory] Registered layer {layer_id} memory: {total_size/(1024**2):.2f} MB with {len(tensors)} tensors")
    
    # try:
    #     print_memory_map()
    # except Exception as e:
    #     print(f"[Memory] Error printing memory map: {e}")
    # print("[Memory] All layers registered successfully")

def org2quant_and_update_kvcache(engine, update_kvcache=True):
    """
    Replace the original layer with the quantized layer and update the kvcache
    """
    start_time = time.perf_counter()
    layer_id = engine.model.next_unquantized_layer
    if layer_id < 0:
        print(f"[Quant Manager] Org2Quant: No more layers to quantize: {layer_id}")
        exit()
    # print(f"[Quant Manager] Org2Quant: Replacing the original layer {engine.model.next_unquantized_layer} to the quantized layer and update kvcache...")
    swap_layer_in_place(engine, layer_id=layer_id)
    time_swapped_original_layer_to_quantized_layer = time.perf_counter()
    # print(f"[Quant Manager] Swapped the original layer to the quantized layer, with time: {(time_swapped_original_layer_to_quantized_layer - start_time)*1000:.2f} ms")
    if update_kvcache:
        update_kvcache_after_replacing_layer(engine, layer_id=layer_id)
    time_updated_kvcache_after_replacing_layer = time.perf_counter()
    # print(f"[Quant Manager] Updated the kvcache after replacing the layer, with time: {(time_updated_kvcache_after_replacing_layer - time_swapped_original_layer_to_quantized_layer)*1000:.2f} ms")
    engine.model.is_layer_quant_list[layer_id] = True
    engine.model.layer_quant_list.append(layer_id)
    engine.model.next_unquantized_layer -= 1
    # time_marked_layer_as_quantized = time.perf_counter()
    # print(f"[Quant Manager: Org2Quant] Marked the layer as quantized, with time: {(time_marked_layer_as_quantized - time_updated_kvcache_after_replacing_layer)*1000:.2f} ms")
    # print(f"******** [Quant_Manager] Org2Quant: Layer {layer_id}, Total replacement time: {(time.perf_counter() - start_time)*1000:.2f} ms.")


def free_layer_weight(layer_weight):
    """
    Delete all tensors contained in the layer weight object and free GPU memory.
    This function assumes that `layer_weight` has attributes that are tensors.
    """
    # print(f"\n[Memory] Deleting registered parameters. ")
    # Delete all registered parameters (if stored in a dictionary-like structure)
    if hasattr(layer_weight, 'registered_weights'):
        for item in layer_weight.registered_weights:
            # Assume each item has an attribute name in 'attr_name'
            tensor = getattr(layer_weight, item.attr_name, None)
            if tensor is not None:
                # print(f"Deleting tensor for {item.key} with shape {tensor.shape}")
                del tensor  # Remove reference to the tensor
            # else:
            #     print(f"[Memory] Warning: Tensor for {item.key} is None")
    
    # print(f"\n[Memory] Deleting parameters and buffers directly. ")
    # Optionally, delete all parameters and buffers directly if they're stored in __dict__
    if hasattr(layer_weight, '__dict__'):
        keys = list(layer_weight.__dict__.keys())
        for key in keys:
            attr = layer_weight.__dict__[key]
            # If the attribute is a tensor and on GPU, delete it.
            if isinstance(attr, torch.Tensor) and attr.device.type == 'cuda':
                # print(f"Deleting attribute {key} with shape {attr.shape}")
                del layer_weight.__dict__[key]
            # else:
            #     print(f"[Memory] Warning: Attribute {key} is not a tensor")
    
    # Delete the layer weight object itself
    del layer_weight

    # Force Python garbage collection
    # gc.collect()

    # Tell PyTorch's caching allocator to release unused memory
    # torch.cuda.empty_cache()
    # print("[Memory] Cleared layer weight and emptied CUDA cache.")

@torch.inference_mode()
def swap_layer_in_place(engine, layer_id):
    """Swap layer using the C++ extension for precise memory control"""
    # Timer for overall process
    # total_start_time = time.perf_counter()
    # print("\n" + "="*30, f"Swapping layer {layer_id} in place", "="*30)

    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # initial_memory = torch.cuda.memory_allocated()
    # print(f"[Engine] Initial GPU memory: {initial_memory / (1024**2):.2f} MB")

    # print(f"[Engine] Free memory before replacement. ")
    # print(f"[Engine] Original layer: {engine.model.transformer_layers[layer_id]}")
    # print(f"[Engine] Original layer vars: {vars(engine.model.transformer_layers[layer_id])}")
    # print(f"[Engine] Original layer weight: {engine.model.transformer_layers[layer_id].weight}")
    # print(f"[Engine] Original layer weight vars: {vars(engine.model.transformer_layers[layer_id].weight)}")
    
    free_layer_weight(engine.model.transformer_layers[layer_id].weight)
    # free_layer_end = time.perf_counter()

    # print(f"\n[Engine] ~~~~~~~ memory after deleting original layer. ~~~~~~~")
    # print(f"[Engine] Original layer: {engine.model.transformer_layers[layer_id]}")
    # print(f"[Engine] Original layer vars: {vars(engine.model.transformer_layers[layer_id])}")
    # print(f"[Engine] Original layer weight: {engine.model.transformer_layers[layer_id].weight}")
    # print(f"[Engine] type of Original layer weight: {type(engine.model.transformer_layers[layer_id].weight)}")
    # if hasattr(engine.model.transformer_layers[layer_id].weight, '__dict__'):
    #     print(f"[Engine] Original layer weight has __dict__: {vars(engine.model.transformer_layers[layer_id].weight)}")
    # else:
    #     print(f"[Engine] This variable does NOT have a __dict__ attribute.")   

    # Replace with None to break references
    # del engine.model.transformer_layers[layer_id]
    # engine.model.transformer_layers[layer_id] = 0
    
    # Force garbage collection and clear CUDA cache
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # memory_after_freeing_org_layer = torch.cuda.memory_allocated()
    # print(f"[Engine] Final GPU memory: {memory_after_freeing_org_layer / (1024**2):.2f} MB")
    # print(f"[Engine] GPU memory increase after freeing original layer: {(memory_after_freeing_org_layer - initial_memory) / (1024**2):.2f} MB")

    # print(f"\n[Engine] ~~~~~~~ memory after freeing original layer. ~~~~~~~")
    # print(f"[Engine] Original layer: {engine.model.transformer_layers[layer_id]}")
    # print(f"[Engine] transformer_layers: {engine.model.transformer_layers}")
    # print(f"[Engine] transformer_layers length: {len(engine.model.transformer_layers)}")
    # print(f"[Engine] Original layer vars: {vars(engine.model.transformer_layers[layer_id])}")

    # Timer for in-place replacement
    # Replace in-place with C++ extension
    result_tensors = replace_layer_org2quant(layer_id)
    # replace_end = time.perf_counter()
    
    # print(f"\n~~~~~~~[Engine] result_tensors ~~~~~~~")
    # print(f"[Engine] result_tensors length: {len(result_tensors)}")
    # for i, tensor in enumerate(result_tensors):
    #     print(f"[Engine] tensor {i}: {tensor}")
    
    # Track initial memory state
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # memory_after_replacing_layer = torch.cuda.memory_allocated()
    # print(f"[Engine] Initial GPU memory: {memory_after_replacing_layer / (1024**2):.2f} MB")
    # print(f"[Engine] GPU memory increase after replacing layer: {(memory_after_replacing_layer - memory_after_freeing_org_layer) / (1024**2):.2f} MB")

    # Layer reconstruction
    # Create HF-compatible config for rebuilding layer
    hf_config = create_hf_compatible_config(engine)
    # new_LlamaDecoderLayer_initial_time = time.perf_counter()
    # Reconstruct LlamaDecoderLayer with the in-place tensors
    with init_empty_weights():
        rebuilt_layer = LlamaDecoderLayer(hf_config, layer_id)
    # new_LlamaDecoderLayer_end_time = time.perf_counter()
    # new_LlamaDecoderLayer_time = (new_LlamaDecoderLayer_end_time - new_LlamaDecoderLayer_initial_time) * 1000
    # print(f"[Engine] Reconstructed LlamaDecoderLayer with {layer_id} in {new_LlamaDecoderLayer_time:.2f} ms")
    
    awq_feature_dimensions = engine.awq_layer_manager.feature_map['dimensions']
    # print(f"[Engine] AWQ feature dimensions: {awq_feature_dimensions}")
    # print(f"[Engine] q_proj: {awq_feature_dimensions['self_attn.q_proj']['in_features']}, {awq_feature_dimensions['self_attn.q_proj']['out_features']}")
    # print(f"[Engine] gate_proj: {awq_feature_dimensions['mlp.gate_proj']['in_features']}, {awq_feature_dimensions['mlp.gate_proj']['out_features']}")
    # Keep track of tensor index
    tensor_idx = 0
    # new_layernorm_weight_initial_time = time.perf_counter()
    # Set layer norm weights
    rebuilt_layer.input_layernorm.weight = nn.Parameter(result_tensors[tensor_idx])
    tensor_idx += 1
    rebuilt_layer.post_attention_layernorm.weight = nn.Parameter(result_tensors[tensor_idx])
    tensor_idx += 1
    # new_layernorm_weight_end_time = time.perf_counter()
    # new_layernorm_weight_time = (new_layernorm_weight_end_time - new_layernorm_weight_initial_time) * 1000
    # print(f"[Engine] Set layer norm weights for {layer_id} in {new_layernorm_weight_time:.2f} ms")
    
    # Replace attention modules with WQLinear
    # new_attention_initial_time = time.perf_counter()
    for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:        
        # Create new WQLinear module
        new_module = WQLinear(
            w_bit=engine.engine_config.quantized_q_config["w_bit"],
            group_size=engine.engine_config.quantized_q_config["q_group_size"],
            in_features=awq_feature_dimensions[f'self_attn.{module_name}']['in_features'],
            out_features=awq_feature_dimensions[f'self_attn.{module_name}']['out_features'],
            bias=False,
            dev="cuda",
            qweight=result_tensors[tensor_idx],
            scales=result_tensors[tensor_idx + 1],
            scaled_zeros=result_tensors[tensor_idx + 2]
        )
        tensor_idx += 3
        
        # Set tensors
        # new_module.qweight = result_tensors[tensor_idx]
        # tensor_idx += 1
        # new_module.scales = result_tensors[tensor_idx]
        # tensor_idx += 1
        # new_module.scaled_zeros = result_tensors[tensor_idx]
        # tensor_idx += 1
        
        # Replace in layer
        setattr(rebuilt_layer.self_attn, module_name, new_module)
    # new_attention_end_time = time.perf_counter()
    # new_attention_time = (new_attention_end_time - new_attention_initial_time) * 1000
    # print(f"[Engine] Replaced attention modules for {layer_id} in {new_attention_time:.2f} ms")
    
    # new_rotary_embedding_initial_time = time.perf_counter()
    # Set rotary embedding if it exists
    # if hasattr(awq_layer.self_attn, 'rotary_emb') and hasattr(awq_layer.self_attn.rotary_emb, 'inv_freq'):
    #     rebuilt_layer.self_attn.rotary_emb.inv_freq = result_tensors[tensor_idx]
    #     tensor_idx += 1
    # new_rotary_embedding_end_time = time.perf_counter()
    # new_rotary_embedding_time = (new_rotary_embedding_end_time - new_rotary_embedding_initial_time) * 1000
    # print(f"[Engine] Set rotary embedding for {layer_id} in {new_rotary_embedding_time:.2f} ms")
    # Replace MLP modules with WQLinear
    # new_mlp_initial_time = time.perf_counter()
    for module_name in ['gate_proj', 'up_proj', 'down_proj']:
        # Create new WQLinear module
        new_module = WQLinear(
            w_bit=engine.engine_config.quantized_q_config["w_bit"],
            group_size=engine.engine_config.quantized_q_config["q_group_size"],
            in_features=awq_feature_dimensions[f'mlp.{module_name}']['in_features'],
            out_features=awq_feature_dimensions[f'mlp.{module_name}']['out_features'],
            bias=False,
            dev="cuda",
            qweight=result_tensors[tensor_idx],
            scales=result_tensors[tensor_idx + 1],
            scaled_zeros=result_tensors[tensor_idx + 2]
        )
        tensor_idx += 3
        # Set tensors
        # new_module.qweight = result_tensors[tensor_idx]
        # tensor_idx += 1
        # new_module.scales = result_tensors[tensor_idx]
        # tensor_idx += 1
        # new_module.scaled_zeros = result_tensors[tensor_idx]
        # tensor_idx += 1
        # Replace in layer
        setattr(rebuilt_layer.mlp, module_name, new_module)
    # new_mlp_end_time = time.perf_counter()
    # new_mlp_time = (new_mlp_end_time - new_mlp_initial_time) * 1000
    # print(f"[Engine] Replaced MLP modules for {layer_id} in {new_mlp_time:.2f} ms")
    # Set to evaluation mode
    rebuilt_layer.eval()
    # rebuild_end = time.perf_counter()
    
    # Timer for creating AWQ transformer layer wrapper
    gpu_layer = AWQTransformerLayer(
        awq_layer=rebuilt_layer,
        model_config=engine.model_config,
        engine_config=engine.engine_config,
        layer_id=layer_id,
        decoding_piggyback_stream=torch.cuda.Stream()
    )
    # Replace in model
    engine.model.transformer_layers[layer_id] = gpu_layer
    # wrap_end = time.perf_counter()
    
    # Get freed memory size
    # used_size = sum(t.numel() * t.element_size() for t in result_tensors)
    # freed_size = get_freed_memory_size(layer_id, used_size)

    # Timer for cleanup
    # cleanup_start = time.perf_counter()
    
    # Force cleanup of temporary objects
    # del gpu_tensors
    # del tensors
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # cleanup_end = time.perf_counter()
    # cleanup_time = (cleanup_end - cleanup_start) * 1000
    # print(f"[Engine] Memory cleanup completed in {cleanup_time:.2f} ms")

    # Report results
    # torch.cuda.synchronize()
    # final_memory = torch.cuda.memory_allocated()
    # memory_change = final_memory - initial_memory
    
    # total_end_time = time.perf_counter()
    # print(f"\n[Memory] === Layer {layer_id} Replacement Summary ===")
    # print(f"[Engine] Free layer weight in {(free_layer_end - total_start_time)*1000:.2f} ms")
    # print(f"[Engine] Replace layer in     {(replace_end - free_layer_end)*1000:.2f} ms")
    # print(f"[Engine] Reconstruct layer in {(rebuild_end - replace_end)*1000:.2f} ms")
    # print(f"[Engine] Wrap layer in        {(wrap_end - rebuild_end)*1000:.2f} ms")
    # print(f"[Quantization] Total quantization time:  {(total_end_time - total_start_time)*1000:.2f} ms: free_layer_end: {(free_layer_end - total_start_time)*1000:.2f} ms, replace_end: {(replace_end - free_layer_end)*1000:.2f} ms, rebuild_end: {(rebuild_end - replace_end)*1000:.2f} ms, wrap_end: {(wrap_end - rebuild_end)*1000:.2f} ms")
    
    # return gpu_layer, freed_size
    # return 0, 0

@torch.inference_mode()
def update_kvcache_after_replacing_layer(engine, layer_id):
    """Acquire new kvcache by replacing the original layer with the smaller quantized layer
       and then acquire the kv tensors from the new available memory"""
    # start_time = time.perf_counter()
    (k_cache_new, v_cache_new) = acquire_new_kvcache(layer_id)
    # print("~"*50, "update kvcache after layer replacement", "~"*50)
    # print(f"[Memory] k_cache_new.shape: {k_cache_new.shape}")
    # print(f"[Memory] v_cache_new.shape: {v_cache_new.shape}")
    # print(f"[Memory] existing k_cache.shape: {engine.model.k_cache.shape}")
    # print(f"[Memory] existing v_cache.shape: {engine.model.v_cache.shape}")
    
    # Canoot use torch.cat() to attach the new kvcache to existing KV caches
    # because it needs to allocate new memory for the total size of the updated KV caches
    # and copy all the data from the old KV caches to the new memory

    # Attach the new kvcache to existing KV caches
    # engine.model.k_cache = torch.cat((engine.model.k_cache, k_cache_new), dim=0)
    # engine.model.v_cache = torch.cat((engine.model.v_cache, v_cache_new), dim=0)

    engine.model.k_cache_new.append(k_cache_new)
    engine.model.v_cache_new.append(v_cache_new)
    # print(f"[Memory] len(engine.model.k_cache_new): {len(engine.model.k_cache_new)}")
    # print(f"[Memory] len(engine.model.v_cache_new): {len(engine.model.v_cache_new)}")
    # print(f"\n[Memory] original engine.model.gpu_block_manager: \n \
    #         device_name: {engine.model.gpu_block_manager.device_name}\n \
    #         num_free_blocks: {engine.model.gpu_block_manager.num_free_blocks}\n \
    #         num_blocks: {engine.model.gpu_block_manager.num_blocks}\n \
    #         block_size: {engine.model.gpu_block_manager.block_size}\n \
    #         num_seq_allocated_blocks shape: {engine.model.gpu_block_manager.num_seq_allocated_blocks.shape}\n \
    #         block_table shape: {engine.model.gpu_block_manager.block_table.shape}\n \
    #         is_block_free shape: {engine.model.gpu_block_manager.is_block_free.shape}")
    org_num_blocks = engine.model.gpu_block_manager.num_blocks
    # org_num_free_blocks = engine.model.gpu_block_manager.num_free_blocks.item()
    org_is_block_free_shape = engine.model.gpu_block_manager.is_block_free.shape

    engine.model.gpu_block_manager.num_blocks += len(k_cache_new)
    engine.model.gpu_block_manager.num_free_blocks += len(k_cache_new)
    engine.model.kv_cache_new_block_size = k_cache_new.shape[0]
    # print(f"\n[Memory] updating vars(engine.model.gpu_block_manager): \n \
    #         device_name: {engine.model.gpu_block_manager.device_name}\n \
    #         num_free_blocks: {engine.model.gpu_block_manager.num_free_blocks}\n \
    #         num_blocks: {engine.model.gpu_block_manager.num_blocks}\n \
    #         block_size: {engine.model.gpu_block_manager.block_size}\n \
    #         num_seq_allocated_blocks shape: {engine.model.gpu_block_manager.num_seq_allocated_blocks.shape}\n \
    #         block_table shape: {engine.model.gpu_block_manager.block_table.shape}\n \
    #         is_block_free shape: {engine.model.gpu_block_manager.is_block_free.shape}")
    is_block_free_new = torch.ones((len(k_cache_new),), dtype=torch.bool, device="cuda")
    engine.model.gpu_block_manager.is_block_free = torch.cat((engine.model.gpu_block_manager.is_block_free, is_block_free_new), dim=0)

    org_num_gpu_blocks_scheduler = engine.scheduler.num_gpu_blocks
    engine.scheduler.num_gpu_blocks += len(k_cache_new)
    

    # print(f"\n[Memory] updated vars(engine.model.gpu_block_manager): \n \
    #         device_name: {engine.model.gpu_block_manager.device_name}\n \
    #         num_free_blocks: {engine.model.gpu_block_manager.num_free_blocks}\n \
    #         num_blocks: {engine.model.gpu_block_manager.num_blocks}\n \
    #         block_size: {engine.model.gpu_block_manager.block_size}\n \
    #         num_seq_allocated_blocks shape: {engine.model.gpu_block_manager.num_seq_allocated_blocks.shape}\n \
    #         block_table shape: {engine.model.gpu_block_manager.block_table.shape}\n \
    #         is_block_free shape: {engine.model.gpu_block_manager.is_block_free.shape}")
    
    # end_time = time.perf_counter()
    # print(f"[KV update] engine.model.gpu_block_manager.num_blocks: {engine.model.gpu_block_manager.num_blocks} (org: {org_num_blocks})")
    # print(f"[KV update] engine.model.gpu_block_manager.num_free_blocks: {engine.model.gpu_block_manager.num_free_blocks} (org: {org_num_free_blocks})")
    # print(f"[KV update] engine.model.gpu_block_manager.is_block_free.shape: {engine.model.gpu_block_manager.is_block_free.shape} (org: {org_is_block_free_shape})")
    # print(f"[KV update] engine.model.kv_cache_new_block_size: {engine.model.kv_cache_new_block_size}")
    # print(f"[KV update] engine.scheduler.num_gpu_blocks: {engine.scheduler.num_gpu_blocks} (org: {org_num_gpu_blocks_scheduler})")
    # print(f"[KV update] Total time to update kvcache after layer replacement: {(end_time - start_time)*1000:.2f} ms")


@torch.inference_mode()
def create_hf_compatible_config(engine):
    """Create a HuggingFace-compatible LlamaConfig from model_config"""
    from transformers import LlamaConfig
    
    # Create compatible config with all required parameters
    hf_config = LlamaConfig(
        hidden_size=engine.model_config.hidden_size,
        intermediate_size=engine.model_config.ffn_inter_dim,
        num_hidden_layers=engine.model_config.num_layers,
        num_attention_heads=engine.model_config.num_q_heads,
        num_key_value_heads=engine.model_config.num_kv_heads,
        hidden_act="silu",
        max_position_embeddings=engine.model_config.max_position_embeddings,
        rms_norm_eps=engine.model_config.rms_norm_eps,
        rope_theta=getattr(engine.model_config, 'rope_theta', 10000.0),
        _attn_implementation="sdpa"  # Required for newer HF versions
    )
    return hf_config



@torch.inference_mode()
def rebuild_original_layer(engine, layer_id, result_tensors):
    """Rebuild the original LlamaTransformerLayer from the restored tensors"""
    # start_time = time.perf_counter()
    
    # Create a new stream for the layer
    decoding_piggyback_stream = torch.cuda.Stream()
    
    # Create a weight container object
    layer_weight = LlamaTransformerLayerWeight(
        layer_id=layer_id,
        model_config=engine.model_config,
        dtype=torch.float16
    )
    
    # Based on the logs, set the tensor attributes directly to match
    # the LlamaTransformerLayerWeight structure
    # Tensor order based on the logs:
    # 0: attn_norm - input_layernorm.weight
    # 1: q_proj - self_attn.q_proj.weight
    # 2: k_proj - self_attn.k_proj.weight
    # 3: v_proj - self_attn.v_proj.weight
    # 4: o_proj - self_attn.o_proj.weight
    # 5: ffn_norm - post_attention_layernorm.weight
    # 6: up_gate_proj - combined up_proj and gate_proj
    # 7: down_proj - mlp.down_proj.weight
    
    # Map tensors to their attributes
    layer_weight.attn_norm = result_tensors[0]
    layer_weight.q_proj = result_tensors[1]
    layer_weight.k_proj = result_tensors[2]
    layer_weight.v_proj = result_tensors[3]
    layer_weight.o_proj = result_tensors[4]
    layer_weight.ffn_norm = result_tensors[5]
    
    # Handle the up_gate_proj tensor by splitting it into up_proj and gate_proj
    # Based on the logs, this is a single tensor of shape [22016, 4096]
    # We need to split it into two tensors of shape [11008, 4096] each
    # up_gate_proj = result_tensors[6]
    # half_size = up_gate_proj.shape[0] // 2
    
    # Split the tensor
    # layer_weight.up_proj = up_gate_proj[:half_size]
    # layer_weight.gate_proj = up_gate_proj[half_size:]
    
    # Set the combined view as well
    layer_weight.up_gate_proj = result_tensors[6]
    
    # Set down_proj
    layer_weight.down_proj = result_tensors[7]
    
    # Create and initialize the LlamaTransformerLayer
    rebuilt_layer = LlamaTransformerLayer(
        engine.model_config,
        engine.engine_config,
        layer_weight,  # Pass the weight object we created
        decoding_piggyback_stream,
        layer_id
    )
    
    # end_time = time.perf_counter()
    # print(f"[Engine] Rebuilt original layer {layer_id} in {(end_time - start_time)*1000:.2f} ms")
    
    return rebuilt_layer

@torch.inference_mode()
def if_kv_cache_unused(engine, layer_id):
    """Check if the KV cache with the given layer_id is unused"""
    start_time = time.perf_counter()
    kv_cache_index = len(engine.model.k_cache_new)-1
    assert kv_cache_index >= 0, f"[Engine] Layer {layer_id} is not quantized, no KV cache tensors to free"
    blocks_to_remove = engine.model.kv_cache_new_block_size
    start_idx_kv_cache = engine.model.gpu_block_manager.num_blocks_org + (blocks_to_remove * kv_cache_index)
    end_idx_kv_cache = start_idx_kv_cache + blocks_to_remove
    if engine.model.gpu_block_manager.is_block_free[start_idx_kv_cache:end_idx_kv_cache].all():
        # print(f"[Engine] KV cache with layer_id {layer_id} is unused")
        # print(f"[Engine] Time to check if KV cache is unused for layer_id {layer_id}: {(time.perf_counter() - start_time)*1000:.2f} ms")
        return True
    else:
        # print(f"[Engine] KV cache with layer_id {layer_id} is used")
        # print(f"[Engine] Time to check if KV cache is unused for layer_id {layer_id}: {(time.perf_counter() - start_time)*1000:.2f} ms")
        return False

def free_quantized_layer(engine, layer_id):
    """Free the AWQ quantized layer from GPU memory"""
    # start_time = time.perf_counter()
    
    # Get the current layer
    layer = engine.model.transformer_layers[layer_id]
    
    # Check if it's an AWQ layer
    if not isinstance(layer, AWQTransformerLayer):
        raise TypeError(f"Layer {layer_id} is not an AWQ layer, cannot restore original")
    
    # Free AWQ layer resources
    if hasattr(layer, 'awq_layer'):
        # Free attention modules
        for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(layer.awq_layer.self_attn, module_name):
                module = getattr(layer.awq_layer.self_attn, module_name)
                if hasattr(module, 'qweight'):
                    del module.qweight
                if hasattr(module, 'scales'):
                    del module.scales
                if hasattr(module, 'scaled_zeros'):
                    del module.scaled_zeros
                
        # Free MLP modules
        for module_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.awq_layer.mlp, module_name):
                module = getattr(layer.awq_layer.mlp, module_name)
                if hasattr(module, 'qweight'):
                    del module.qweight
                if hasattr(module, 'scales'):
                    del module.scales
                if hasattr(module, 'scaled_zeros'):
                    del module.scaled_zeros
        
        # Free layernorm weights
        if hasattr(layer.awq_layer, 'input_layernorm') and hasattr(layer.awq_layer.input_layernorm, 'weight'):
            del layer.awq_layer.input_layernorm.weight
        if hasattr(layer.awq_layer, 'post_attention_layernorm') and hasattr(layer.awq_layer.post_attention_layernorm, 'weight'):
            del layer.awq_layer.post_attention_layernorm.weight
        
        # Free the AWQ layer itself
        del layer.awq_layer
    
    # Break other references
    # engine.model.transformer_layers[layer_id] = None
    
    # Force cleanup
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # end_time = time.perf_counter()
    # print(f"[Engine] Freed quantized layer in {(end_time - start_time)*1000:.2f} ms")
    

@torch.inference_mode()
def free_kv_cache_tensors(engine, layer_id):
    """Free any KV cache tensors that were allocated in the freed space"""
    # start_time = time.perf_counter()
    
    kv_cache_index = len(engine.model.k_cache_new)-1
    assert kv_cache_index >= 0, f"[Engine] Layer {layer_id} is not quantized, no KV cache tensors to free"
    
    del engine.model.k_cache_new[kv_cache_index]
    del engine.model.v_cache_new[kv_cache_index]
    
    # end_time = time.perf_counter()
    # print(f"[Engine] Freed KV cache tensors in {(end_time - start_time)*1000:.2f} ms")
    

@torch.inference_mode()
def quant2org_and_update_kvcache(engine, update_kvcache=True):
    """Restore original layer by replacing quantized layer and freeing KV cache tensors"""
    total_start_time = time.perf_counter()
    layer_id = engine.model.next_unquantized_layer+1
    
    # Step 0: Check if the KV cache is unused
    if update_kvcache and not if_kv_cache_unused(engine, layer_id):
        return False
    # print(f"[Engine] Restoring layer {layer_id} to original state...")
    
    # Step 1: Free the AWQ quantized layer
    free_quantized_layer(engine, layer_id)
    
    # Step 2: Free the allocated KV cache tensors if they exist
    if update_kvcache:
        free_kv_cache_tensors(engine, layer_id)
    
    # Step 3: Copy the original layer back to GPU using C++ helper
    result_tensors = replace_layer_quant2org(layer_id)
    
    # Step 4: Reconstruct the LlamaTransformerLayer with the original tensors
    rebuilt_layer = rebuild_original_layer(engine, layer_id, result_tensors)
    
    # Step 5: Update the model with the reconstructed layer
    engine.model.transformer_layers[layer_id] = rebuilt_layer
    
    # Step 6: Update the tracking state
    org_num_blocks = engine.model.gpu_block_manager.num_blocks
    org_num_free_blocks = engine.model.gpu_block_manager.num_free_blocks
    org_is_block_free_shape = engine.model.gpu_block_manager.is_block_free.shape
    org_num_gpu_blocks_scheduler = engine.scheduler.num_gpu_blocks
    engine.model.gpu_block_manager.num_blocks -= engine.model.kv_cache_new_block_size
    engine.model.gpu_block_manager.num_free_blocks -= engine.model.kv_cache_new_block_size
    engine.model.gpu_block_manager.is_block_free = engine.model.gpu_block_manager.is_block_free[:-engine.model.kv_cache_new_block_size]
    engine.scheduler.num_gpu_blocks -= engine.model.kv_cache_new_block_size

    engine.model.is_layer_quant_list[layer_id] = False
    engine.model.layer_quant_list.remove(layer_id)
    engine.model.next_unquantized_layer = layer_id

    # print(f"[Engine] engine.model.gpu_block_manager.num_blocks: {engine.model.gpu_block_manager.num_blocks} (org: {org_num_blocks})")
    # print(f"[Engine] engine.model.gpu_block_manager.num_free_blocks: {engine.model.gpu_block_manager.num_free_blocks} (org: {org_num_free_blocks})")
    # print(f"[Engine] engine.model.gpu_block_manager.is_block_free.shape: {engine.model.gpu_block_manager.is_block_free.shape} (org: {org_is_block_free_shape})")
    # print(f"[Engine] engine.scheduler.num_gpu_blocks: {engine.scheduler.num_gpu_blocks} (org: {org_num_gpu_blocks_scheduler})")
    # print(f"******** [Quant_Manager] Quant2Org: Layer {layer_id}, Total restoration time: {(time.perf_counter() - total_start_time)*1000:.2f} ms.")
    return True