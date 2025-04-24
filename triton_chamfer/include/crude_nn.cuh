#ifndef CRUDE_NN_CUH
#define CRUDE_NN_CUH

#include <c10/core/DeviceType.h>
#include <cstdint>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, uint32_t N_POS_DIMS, tcnn::HashType HASH_TYPE>
__global__ void HashInputKernel(const uint32_t num_elements,
                                const tcnn::GridOffsetTable offset_table,
                                const uint32_t base_resolution,
                                const float log2_per_level_scale,
                                T *__restrict__ grid,
                                tcnn::MatrixView<const T> positions_in) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements)
    return;

  const uint32_t level = blockIdx.y; // <- the level is the same for all threads
  grid += offset_table.data[level] * N_POS_DIMS;
  const uint32_t hashmap_size =
      offset_table.data[level + 1] - offset_table.data[level];

  const float scale =
      tcnn::grid_scale(level, log2_per_level_scale, base_resolution);
  const uint32_t resolution = tcnn::grid_resolution(scale);

  T pos[N_POS_DIMS];
  tcnn::uvec<N_POS_DIMS> pos_grid;

  TCNN_PRAGMA_UNROLL
  for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
    tcnn::pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
                    tcnn::identity_fun);
  }

  const uint32_t hash_index = tcnn::grid_index<N_POS_DIMS, HASH_TYPE>(
      tcnn::GridType::Hash, hashmap_size, resolution, pos_grid);

  TCNN_PRAGMA_UNROLL
  for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
    grid[hash_index * N_POS_DIMS + dim] = positions_in(dim, i);
  }
}

template <typename T, uint32_t N_POS_DIMS, tcnn::HashType HASH_TYPE>
__global__ void QueryDistanceKernel(
    const uint32_t num_elements, const uint32_t n_levels,
    const tcnn::GridOffsetTable offset_table, const uint32_t base_resolution,
    const float log2_per_level_scale, T *__restrict__ grid,
    tcnn::MatrixView<const T> positions_query, T *__restrict__ dists) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements)
    return;

  for (uint32_t level = 0; level < n_levels; level++) {
    grid += offset_table.data[level] * N_POS_DIMS;

    const uint32_t hashmap_size =
        offset_table.data[level + 1] - offset_table.data[level];

    const float scale =
        tcnn::grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = tcnn::grid_resolution(scale);

    T pos[N_POS_DIMS];
    tcnn::uvec<N_POS_DIMS> pos_grid;

    TCNN_PRAGMA_UNROLL
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      tcnn::pos_fract(positions_query(dim, i), &pos[dim], &pos_grid[dim], scale,
                      tcnn::identity_fun);
    }

    const uint32_t hash_index = tcnn::grid_index<N_POS_DIMS, HASH_TYPE>(
        tcnn::GridType::Hash, hashmap_size, resolution, pos_grid);

    tcnn::tvec<T, N_POS_DIMS> coord;
    bool valid = true;
    TCNN_PRAGMA_UNROLL
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      T val = grid[hash_index * N_POS_DIMS + dim];
      if (val == INFINITY) {
        valid = false;
        break;
      }
      coord[dim] = val;
    }

    // Break double for loop
    if (!valid)
      break;

    T dist2 = 0;
    TCNN_PRAGMA_UNROLL
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      dist2 += std::pow(coord[dim] - positions_query(dim, i), 2);
    }
    dists[i] = std::sqrt(dist2);
  }
}

template <typename T = float, uint32_t N_POS_DIMS = 3,
          tcnn::HashType HASH_TYPE = tcnn::HashType::CoherentPrime,
          uint32_t N_THREADS_HASHGRID = 512>
torch::Tensor CrudeNN(cudaStream_t stream, const torch::Tensor &input,
                      const torch::Tensor &query) {
  uint32_t n_input = input.size(0);

  // TODO: Expose all configs
  uint32_t log2_hashmap_size = 19;
  uint32_t n_levels = 8;
  // We start at 1 so it is guaranteed nonzero
  uint32_t base_resolution = 1;
  // Golden ratio
  float per_level_scale = 1.618;

  tcnn::GridOffsetTable offset_table;

  uint32_t offset = 0;

  for (uint32_t i = 0; i < n_levels; ++i) {
    // Compute number of dense params required for the given level
    const float scale =
        tcnn::grid_scale(i, std::log2(per_level_scale), base_resolution);
    const uint32_t resolution = tcnn::grid_resolution(scale);

    uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
    uint32_t params_in_level =
        std::pow((float)resolution, N_POS_DIMS) > (float)max_params
            ? max_params
            : tcnn::powi(resolution, N_POS_DIMS);

    // Make sure memory accesses will be aligned
    params_in_level = tcnn::next_multiple(params_in_level, 8u);

    // If hash table needs fewer params than dense, then use fewer and rely on
    // the hash.
    params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));

    offset_table.data[i] = offset;
    offset += params_in_level;
  }

  offset_table.data[n_levels] = offset;
  offset_table.size = n_levels + 1;

  uint32_t n_params = offset_table.data[n_levels] * N_POS_DIMS;

  // Initialize the hashgrid from the GPU, because the number of parameters can
  // be quite large.
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor params = INFINITY * torch::ones({n_params}, opts);

  // Idea: each block only takes care of _one_ hash level (but may iterate over
  // multiple input elements). This way, only one level of the hashmap needs to
  // fit into caches at a time (and it reused for consecutive elements) until it
  // is time to process the next level.

  const dim3 blocks_hashgrid = {tcnn::div_round_up(n_input, N_THREADS_HASHGRID),
                                n_levels, 1};

  tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor> input_matrix(
      input.data_ptr<T>(), N_POS_DIMS, n_input);

  HashInputKernel<T, N_POS_DIMS, HASH_TYPE>
      <<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
          n_input, offset_table, base_resolution, std::log2(per_level_scale),
          params.data_ptr<T>(), input_matrix.view());

  uint32_t n_query = query.size(0);
  tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor> query_matrix(
      query.data_ptr<T>(), N_POS_DIMS, n_query);
  torch::Tensor Da = torch::zeros({n_query}, opts);

  const dim3 blocks_query = {tcnn::div_round_up(n_query, N_THREADS_HASHGRID), 1,
                             1};

  QueryDistanceKernel<T, N_POS_DIMS, HASH_TYPE>
      <<<blocks_query, N_THREADS_HASHGRID, 0, stream>>>(
          n_query, n_levels, offset_table, base_resolution,
          std::log2(per_level_scale), params.data_ptr<T>(), query_matrix.view(),
          Da.data_ptr<T>());

  return Da;
}
#endif // CRUDE_NN_CUH