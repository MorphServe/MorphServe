#pragma once

#include <torch/extension.h>

#include <vector>
#include <cstdint>

void swap_blocks(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,

	torch::Tensor k_cache,
	torch::Tensor v_cache,
	torch::Tensor k_swap,
	torch::Tensor v_swap
);

void swap_blocks_new_kv_combined(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,
	const int64_t num_blocks_org,
	const int64_t org_layer_param_size,
	const int64_t kv_cache_new_block_size,
	const int64_t k_cache_element_size,
	
	torch::Tensor k_cache,
	torch::Tensor v_cache,
	torch::Tensor k_swap,
	torch::Tensor v_swap,
	torch::Tensor k_cache_new,
	torch::Tensor v_cache_new
);