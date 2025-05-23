// memory_manager.h
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace swiftllm {

// Structure to track memory regions
struct MemoryRange {
    void* start_addr;
    size_t size;
};

struct KVCacheInfo {
    int64_t num_layers;
    int64_t num_kv_heads;
    int64_t block_size;
    int64_t head_dim;
};

class MemoryManager {
public:
    // Register a layer's memory range
    static void register_layer_memory_org_gpu(int layer_id, int64_t start_addr_int, size_t size);
    static void register_layer_memory_org_cpu(int layer_id, int64_t start_addr_int, size_t size);
    static void register_layer_memory_quant(int layer_id, int64_t start_addr_int, size_t size);
    static void register_layer_memory_tensor_map_quant(int layer_id, const std::vector<std::map<std::string, std::map<std::string, py::object>>>& tensor_map);
    static void register_layer_memory_tensor_map_org(int layer_id, const std::vector<std::map<std::string, std::map<std::string, py::object>>>& tensor_map);
    
    // Register KV cache info
    static void register_kv_cache_info(int64_t num_layers, int64_t num_kv_heads, int64_t block_size, int64_t head_dim);
    
    // Replace layer in-place with new tensors
    static std::vector<torch::Tensor> replace_layer_org2quant(int layer_id);
    static std::vector<torch::Tensor> replace_layer_quant2org(int layer_id);

    // Acquire new KV cache
    static std::tuple<torch::Tensor, torch::Tensor> acquire_new_kvcache(int layer_id);
    
    // Verify memory continuity of tensors
    static bool verify_tensor_continuity(const std::vector<torch::Tensor>& tensors);

    // Get freed memory size (for KV cache)
    static size_t get_freed_memory_size(int layer_id, size_t used_size);

    // Print memory map
    // static void print_memory_map(const std::unordered_map<int, MemoryRange>& memory_map);
    
    // Getter for memory maps
    static const std::unordered_map<int, MemoryRange>& get_layer_memory_map_org_gpu();
    static const std::unordered_map<int, MemoryRange>& get_layer_memory_map_quant();

private:
    // Global map to track memory regions by layer
    static KVCacheInfo kv_cache_info;
    static std::unordered_map<int, MemoryRange> layer_memory_map_org_gpu;
    static std::unordered_map<int, MemoryRange> layer_memory_map_org_cpu;
    static std::unordered_map<int, MemoryRange> layer_memory_map_quant;
    static std::unordered_map<int, std::vector<std::map<std::string, std::map<std::string, py::object>>>> layer_memory_tensor_map_quant;
    static std::unordered_map<int, std::vector<std::map<std::string, std::map<std::string, py::object>>>> layer_memory_tensor_map_org;
};

} // namespace swiftllm