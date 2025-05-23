// memory_manager.cpp
#include "memory_manager.h"
#include <sstream>

namespace swiftllm {

// Initialize static member
KVCacheInfo MemoryManager::kv_cache_info;
std::unordered_map<int, MemoryRange> MemoryManager::layer_memory_map_org_gpu;
std::unordered_map<int, MemoryRange> MemoryManager::layer_memory_map_org_cpu;
std::unordered_map<int, MemoryRange> MemoryManager::layer_memory_map_quant;
std::unordered_map<int, std::vector<std::map<std::string, std::map<std::string, py::object>>>> MemoryManager::layer_memory_tensor_map_quant;
std::unordered_map<int, std::vector<std::map<std::string, std::map<std::string, py::object>>>> MemoryManager::layer_memory_tensor_map_org;


void MemoryManager::register_layer_memory_org_gpu(int layer_id, int64_t start_addr_int, size_t size) {
    void* start_addr = reinterpret_cast<void*>(start_addr_int);
    layer_memory_map_org_gpu[layer_id] = {start_addr, size};
    // std::cout << "[MemoryManager] Registered layer " << layer_id << " at address " 
    //           << start_addr << " with size " << size << " bytes" << std::endl;
}

void MemoryManager::register_layer_memory_org_cpu(int layer_id, int64_t start_addr_int, size_t size) {
    void* start_addr = reinterpret_cast<void*>(start_addr_int);
    layer_memory_map_org_cpu[layer_id] = {start_addr, size};
    // std::cout << "[MemoryManager] Registered layer " << layer_id << " at address " 
    //           << start_addr << " with size " << size << " bytes" << std::endl;
}

void MemoryManager::register_layer_memory_quant(int layer_id, int64_t start_addr_int, size_t size) {
    void* start_addr = reinterpret_cast<void*>(start_addr_int);
    layer_memory_map_quant[layer_id] = {start_addr, size};
    // std::cout << "[MemoryManager] Registered layer " << layer_id << " at address " 
    //           << start_addr << " with size " << size << " bytes" << std::endl;
}

void MemoryManager::register_layer_memory_tensor_map_quant(int layer_id, const std::vector<std::map<std::string, std::map<std::string, py::object>>>& tensor_map) {
    layer_memory_tensor_map_quant[layer_id] = tensor_map;
    // std::cout << "[MemoryManager] Registered layer " << layer_id << " tensor map" << std::endl;
    // for (const auto& tensor_map : layer_memory_tensor_map_quant[layer_id]) {
    //     for (const auto& [name, info] : tensor_map) {
    //         std::cout << "  " << name << ": " << info << std::endl;
    //     }
    // }
}

void MemoryManager::register_layer_memory_tensor_map_org(int layer_id, const std::vector<std::map<std::string, std::map<std::string, py::object>>>& tensor_map) {
    layer_memory_tensor_map_org[layer_id] = tensor_map;
    // std::cout << "[MemoryManager] Registered layer " << layer_id << " tensor map" << std::endl;
    // for (const auto& tensor_map : layer_memory_tensor_map_org[layer_id]) {
    //     for (const auto& [name, info] : tensor_map) {
    //         std::cout << "  " << name << ": " << info << std::endl;
    //     }
    // }
}

void MemoryManager::register_kv_cache_info(int64_t num_layers, int64_t num_kv_heads, int64_t block_size, int64_t head_dim) {
    kv_cache_info.num_layers = num_layers;
    kv_cache_info.num_kv_heads = num_kv_heads;
    kv_cache_info.block_size = block_size;
    kv_cache_info.head_dim = head_dim;
    // std::cout << "[MemoryManager] Registered KV cache info, num_layers: " << kv_cache_info.num_layers
    //           << ", num_kv_heads: " << kv_cache_info.num_kv_heads << ", block_size: " << kv_cache_info.block_size 
    //           << ", head_dim: " << kv_cache_info.head_dim << std::endl;

}


std::tuple<torch::Tensor, torch::Tensor> MemoryManager::acquire_new_kvcache(int layer_id) {
    if (layer_memory_map_org_gpu.find(layer_id) == layer_memory_map_org_gpu.end() ||
        layer_memory_map_quant.find(layer_id) == layer_memory_map_quant.end()) {
        throw std::runtime_error("Layer not found in memory maps");
    }

    // Get memory information
    auto& org_range = layer_memory_map_org_gpu[layer_id];
    auto& quant_range = layer_memory_map_quant[layer_id];
    
    // Calculate new available memory
    // size_t freed_memory = org_range.size - quant_range.size;
    
    // Calculate new starting address for KV cache
    // Align to 256-byte boundary
    char* layer_end = static_cast<char*>(org_range.start_addr) + quant_range.size;
    void* new_kv_start = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(layer_end) + 255) & ~255);

    // Calculate maximum number of new blocks that can fit
    size_t available_memory = org_range.size - (static_cast<char*>(new_kv_start) - 
        static_cast<char*>(org_range.start_addr));
    
    // kv_slot_size = (k+v) * num_layers * num_kv_heads * head_dim * fp16_2bytes
    size_t kv_slot_size = 2 * kv_cache_info.num_layers * kv_cache_info.num_kv_heads * kv_cache_info.head_dim * 2; 
    
    // kv_block_size = kv_slot_size * block_size
    size_t kv_block_size = kv_slot_size * kv_cache_info.block_size;

    size_t max_new_blocks = available_memory / kv_block_size;
    
    if (max_new_blocks == 0) {
        throw std::runtime_error("No space available for new KV cache blocks");
    }
    
    // std::cout << "[MemoryManager] Acquiring new KV cache:" << std::endl
    //             << "  Freed memory: " << freed_memory / (1024*1024) << " MB" << std::endl
    //             << "  Quant layer end: " << reinterpret_cast<void*>(layer_end) << std::endl
    //             << "  Available for KV: " << available_memory / (1024*1024) << " MB" << std::endl
    //             << "  available for KV start: " << reinterpret_cast<void*>(new_kv_start) << std::endl
    //             << "  kv_slot_size: " << kv_slot_size  << " bytes" << std::endl
    //             << "  kv_block_size: " << kv_block_size << " bytes" << std::endl
    //             << "  New blocks possible: " << max_new_blocks << std::endl
    //             << "  New KV cache memory usage: " << max_new_blocks * kv_block_size << " bytes" << std::endl;

    // Create new tensor options
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, 0)
        .layout(torch::kStrided);
    
    // Calculate new shape
    std::vector<int64_t> new_shape = {
        static_cast<int64_t>(max_new_blocks),
        kv_cache_info.num_layers,
        kv_cache_info.num_kv_heads,
        kv_cache_info.block_size,
        kv_cache_info.head_dim
    };
    
    // Create K cache tensor
    auto k_storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        max_new_blocks * (kv_block_size/2),
        c10::DataPtr(new_kv_start, c10::Device(c10::DeviceType::CUDA, 0)),
        nullptr,
        false
    );
    
    auto k_cache = torch::empty({}, options).set_(
        k_storage,
        0,
        new_shape,
        {} // Let PyTorch compute strides
    );
    
    // Create V cache tensor
    void* v_start = static_cast<char*>(new_kv_start) + max_new_blocks * (kv_block_size/2);
    auto v_storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        max_new_blocks * (kv_block_size/2),
        c10::DataPtr(v_start, c10::Device(c10::DeviceType::CUDA, 0)),
        nullptr,
        false
    );
    
    auto v_cache = torch::empty({}, options).set_(
        v_storage,
        0,
        new_shape,
        {} // Let PyTorch compute strides
    );
    
    // Initialize tensors to zero
    k_cache.zero_();
    v_cache.zero_();
    
    // std::cout << "[MemoryManager] Created new KV cache tensors:" << std::endl
    //               << "  Shape: [" << max_new_blocks << ", " << kv_cache_info.num_layers << ", "
    //               << kv_cache_info.num_kv_heads << ", " << kv_cache_info.block_size << ", "
    //               << kv_cache_info.head_dim << "]" << std::endl
    //               << "  K cache address: " << new_kv_start << std::endl
    //               << "  V cache address: " << v_start << std::endl
    //               << "  End of K cache: " << reinterpret_cast<void*>(static_cast<char*>(new_kv_start) + max_new_blocks * (kv_block_size/2)) << std::endl
    //               << "  End of V cache: " << reinterpret_cast<void*>(static_cast<char*>(v_start) + max_new_blocks * (kv_block_size/2)) << std::endl;
        
    return std::make_tuple(k_cache, v_cache);

}


std::vector<torch::Tensor> MemoryManager::replace_layer_org2quant(int layer_id) {
    if (layer_memory_map_org_gpu.find(layer_id) == layer_memory_map_org_gpu.end() or 
        layer_memory_map_quant.find(layer_id) == layer_memory_map_quant.end() or 
        layer_memory_tensor_map_quant.find(layer_id) == layer_memory_tensor_map_quant.end()) {
        std::stringstream ss;
        ss << "Layer ID " << layer_id << " not registered in memory map";
        throw std::runtime_error(ss.str());
    }
    
    auto& mem_range_org = layer_memory_map_org_gpu[layer_id];
    auto& mem_range_quant = layer_memory_map_quant[layer_id];
    void* dst_addr = mem_range_org.start_addr;  // GPU memory address
    void* src_addr = mem_range_quant.start_addr;  // CPU memory address
    size_t available_size = mem_range_org.size;  // GPU memory size
    size_t layer_size = mem_range_quant.size;  // CPU memory size
    
    if (layer_size > available_size) {
        std::stringstream ss;
        ss << "Layer requires " << layer_size << " bytes, but only " 
           << available_size << " bytes available";
        throw std::runtime_error(ss.str());
    } 
    
    // Create CUDA stream for async transfer
    cudaStream_t stream;
    cudaError_t cuda_status = cudaStreamCreate(&stream);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA stream creation failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }

    // Copy memory from CPU to GPU
    cuda_status = cudaMemcpyAsync(
        dst_addr,
        src_addr,
        layer_size,
        cudaMemcpyHostToDevice,
        stream
    );

    
    if (cuda_status != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("CUDA memcpy failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Synchronize to ensure copy is complete
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("CUDA stream synchronize failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }
    
    cudaStreamDestroy(stream);

    // Create result tensors that point to regions within the transferred memory
    std::vector<torch::Tensor> result_tensors;
    
    // std::cout << "[MemoryManager] Reconstructed layer " << layer_id << " tensor map" << std::endl;
    for (const auto& tensor_map : layer_memory_tensor_map_quant[layer_id]) {
        for (const auto& [name, info] : tensor_map) {
            size_t offset = py::cast<size_t>(info.at("offset"));
            std::vector<int64_t> shape = py::cast<std::vector<int64_t>>(info.at("shape"));
            c10::ScalarType dtype = py::cast<c10::ScalarType>(info.at("dtype"));
            size_t size = py::cast<size_t>(info.at("size"));
            // std::cout << "  " << name << ": " << offset << ", " << shape << ", " << dtype << ", " << size << std::endl;
            
            // Create tensor options
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(torch::kCUDA, 0)
                .layout(torch::kStrided);
            
            // Pointer to this tensor within the GPU memory
            void* tensor_addr = static_cast<char*>(dst_addr) + offset;
            
            // Create storage
            auto storage = c10::Storage(
                c10::Storage::use_byte_size_t(),
                size,
                c10::DataPtr(tensor_addr, c10::Device(c10::DeviceType::CUDA, 0)),
                nullptr,
                false
            );
            
            // Create tensor from storage with correct shape
            auto tensor = torch::empty({}, options).set_(
                storage,
                0,
                shape,
                {} // Let PyTorch compute strides
            );
            
            result_tensors.push_back(tensor);
        }
    }
    
    // std::cout << "[MemoryManager] Replaced layer " << layer_id << std::endl
    //           << "  on memory start: " << mem_range_org.start_addr << std::endl
    //           << "  with original layer size: " << mem_range_org.size << std::endl
    //           << "  with memory start: " << mem_range_quant.start_addr << std::endl
    //           << "  with quantized layer size: " << mem_range_quant.size << std::endl
    //           << "  memory available for new KV from address: " << reinterpret_cast<void*>(static_cast<char*>(mem_range_org.start_addr) + mem_range_quant.size) << std::endl
    //           << "  End of available memory: " << reinterpret_cast<void*>(static_cast<char*>(mem_range_org.start_addr) + mem_range_org.size) << std::endl;
    
    return result_tensors;
}


std::vector<torch::Tensor> MemoryManager::replace_layer_quant2org(int layer_id) {
    if (layer_memory_map_org_gpu.find(layer_id) == layer_memory_map_org_gpu.end() or 
        layer_memory_map_org_cpu.find(layer_id) == layer_memory_map_org_cpu.end() or
        layer_memory_tensor_map_org.find(layer_id) == layer_memory_tensor_map_org.end()) {
        std::stringstream ss;
        ss << "[MemoryManager C++] Layer ID " << layer_id << " not registered in original memory maps";
        throw std::runtime_error(ss.str());
    }

    auto& mem_range_gpu = layer_memory_map_org_gpu[layer_id];
    auto& mem_range_cpu = layer_memory_map_org_cpu[layer_id];
    void* dst_addr = mem_range_gpu.start_addr;  // GPU memory address
    void* src_addr = mem_range_cpu.start_addr;  // CPU memory address
    size_t layer_size = mem_range_cpu.size;     // Size of the original layer
    
    // Create CUDA stream for async transfer
    cudaStream_t stream;
    cudaError_t cuda_status = cudaStreamCreate(&stream);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA stream creation failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }

    // Time the memory copy operation
    // auto start_time = std::chrono::high_resolution_clock::now();

    // Copy memory from CPU to GPU
    cuda_status = cudaMemcpyAsync(
        dst_addr,
        src_addr,
        layer_size,
        cudaMemcpyHostToDevice,
        stream
    );
    
    if (cuda_status != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("CUDA memcpy failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }

    // Synchronize to ensure copy is complete
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("CUDA stream synchronize failed: " + 
            std::string(cudaGetErrorString(cuda_status)));
    }
    
    // auto end_time_swap = std::chrono::high_resolution_clock::now();
    // auto duration_swap = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_swap - start_time).count();
    
    // std::cout << "[MemoryManager C++] Swapped original layer " << layer_id << " from CPU to GPU in " 
    //           << duration_swap << " ms" << std::endl
    //           << "  CPU address: " << mem_range_cpu.start_addr << std::endl
    //           << "  GPU address: " << mem_range_gpu.start_addr << std::endl
    //           << "  Size: " << (layer_size / (1024*1024)) << " MB" << std::endl;
    
    cudaStreamDestroy(stream);

    // Create result tensors that point to regions within the transferred memory
    std::vector<torch::Tensor> result_tensors;
    
    // Use the tensor map to reconstruct tensors
    for (const auto& tensor_map : layer_memory_tensor_map_org[layer_id]) {
        for (const auto& [name, info] : tensor_map) {
            size_t offset = py::cast<size_t>(info.at("offset"));
            std::vector<int64_t> shape = py::cast<std::vector<int64_t>>(info.at("shape"));
            c10::ScalarType dtype = py::cast<c10::ScalarType>(info.at("dtype"));
            size_t size = py::cast<size_t>(info.at("size"));
            
            // std::cout << "  Creating tensor for " << name << ": offset=" << offset << ", shape=" << shape 
            //           << ", dtype=" << dtype << ", size=" << size << std::endl;
            
            // Create tensor options
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(torch::kCUDA, 0)
                .layout(torch::kStrided);
            
            // Pointer to this tensor within the GPU memory
            void* tensor_addr = static_cast<char*>(dst_addr) + offset;
            
            // Create storage
            auto storage = c10::Storage(
                c10::Storage::use_byte_size_t(),
                size,
                c10::DataPtr(tensor_addr, c10::Device(c10::DeviceType::CUDA, 0)),
                nullptr,
                false
            );
            
            // Create tensor from storage with correct shape
            auto tensor = torch::empty({}, options).set_(
                storage,
                0,
                shape,
                {} // Let PyTorch compute strides
            );
            
            result_tensors.push_back(tensor);
        }
    }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // std::cout << "[MemoryManager C++] Restore original layer " << layer_id << " from GPU to CPU in " 
    //           << duration << " ms" << std::endl
    //           << "  GPU address: " << mem_range_gpu.start_addr << std::endl
    //           << "  CPU address: " << mem_range_cpu.start_addr << std::endl
    //           << "  Size: " << (layer_size / (1024*1024)) << " MB" << std::endl;

    return result_tensors;
}


size_t MemoryManager::get_freed_memory_size(int layer_id, size_t used_size) {
    if (layer_memory_map_org_gpu.find(layer_id) == layer_memory_map_org_gpu.end()) {
        return 0;
    }
    
    size_t original_size = layer_memory_map_org_gpu[layer_id].size;
    return original_size > used_size ? original_size - used_size : 0;
}

bool MemoryManager::verify_tensor_continuity(const std::vector<torch::Tensor>& tensors) {
    if (tensors.empty()) {
        return true;
    }
    
    // Sort tensors by memory address
    std::vector<std::pair<void*, size_t>> sorted_tensors;
    for (const auto& tensor : tensors) {
        void* data_ptr = tensor.data_ptr();
        size_t size = tensor.numel() * tensor.element_size();
        sorted_tensors.push_back({data_ptr, size});
    }
    
    std::sort(sorted_tensors.begin(), sorted_tensors.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Check for continuity
    bool is_continuous = true;
    for (size_t i = 1; i < sorted_tensors.size(); i++) {
        char* prev_end = static_cast<char*>(sorted_tensors[i-1].first) + sorted_tensors[i-1].second;
        char* curr_start = static_cast<char*>(sorted_tensors[i].first);
        
        if (prev_end != curr_start) {
            std::cout << "[MemoryManager] Gap detected between tensors: " 
                      << (curr_start - prev_end) << " bytes" << std::endl;
            is_continuous = false;
        }
    }
    
    return is_continuous;
}

// void MemoryManager::print_memory_map(const std::unordered_map<int, MemoryRange>& memory_map) {
//     std::cout << "[MemoryManager] Memory Map (" << memory_map.size() << " layers):" << std::endl;
//     for (const auto& [layer_id, range] : memory_map) {
//         std::cout << "  Layer " << layer_id << ": Address 0x" 
//                   << std::hex << reinterpret_cast<uintptr_t>(range.start_addr) << std::dec
//                   << ", Size " << (range.size / (1024.0 * 1024.0)) << " MB" << std::endl;
//     }
// }

// Getter for memory maps
const std::unordered_map<int, MemoryRange>& MemoryManager::get_layer_memory_map_org_gpu() {
    return layer_memory_map_org_gpu;
}

const std::unordered_map<int, MemoryRange>& MemoryManager::get_layer_memory_map_quant() {
    return layer_memory_map_quant;
}


} // namespace swiftllm