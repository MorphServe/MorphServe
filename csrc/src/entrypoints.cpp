#include <torch/extension.h>

#include "block_swapping.h"
#include "memory_manager.h"

PYBIND11_MODULE(swiftllm_c, m) {
  m.def("swap_blocks", &swap_blocks);
  m.def("swap_blocks_new_kv_combined", &swap_blocks_new_kv_combined);
  m.def("register_layer_memory_org_gpu", &swiftllm::MemoryManager::register_layer_memory_org_gpu, 
        "Register layer memory range");
  m.def("register_layer_memory_org_cpu", &swiftllm::MemoryManager::register_layer_memory_org_cpu, 
        "Register layer memory range");
  m.def("register_layer_memory_quant", &swiftllm::MemoryManager::register_layer_memory_quant, 
        "Register layer memory range");
  m.def("register_layer_memory_tensor_map_quant", &swiftllm::MemoryManager::register_layer_memory_tensor_map_quant, 
        "Register layer memory tensor map");
  m.def("register_layer_memory_tensor_map_org", &swiftllm::MemoryManager::register_layer_memory_tensor_map_org, 
        "Register layer memory tensor map");
  m.def("register_kv_cache_info", &swiftllm::MemoryManager::register_kv_cache_info, 
        "Register KV cache info");
  m.def("acquire_new_kvcache", &swiftllm::MemoryManager::acquire_new_kvcache, 
        "Acquire new KV cache");
  m.def("replace_layer_org2quant", &swiftllm::MemoryManager::replace_layer_org2quant, 
        "Replace layer org2quant");
  m.def("replace_layer_quant2org", &swiftllm::MemoryManager::replace_layer_quant2org, 
        "Replace layer quant2org");
  m.def("get_freed_memory_size", &swiftllm::MemoryManager::get_freed_memory_size, 
        "Get freed memory size");
  m.def("verify_tensor_continuity", &swiftllm::MemoryManager::verify_tensor_continuity, 
        "Verify tensors are continuous in memory");
//   m.def("print_memory_map", &swiftllm::MemoryManager::print_memory_map, 
//         "Print memory allocation map");
}

