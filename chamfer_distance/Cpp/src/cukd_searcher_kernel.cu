#include "cukd_searcher.h"

#include <ATen/cuda/CUDAContext.h>
#include <omp.h>

#define CUKD_BUILDER_INPLACE
#include <cukd/builder.h>
#include <cukd/fcp.h>

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

template <typename PointT>
struct OrderedPoint_traits : public cukd::default_data_traits<PointT> {
  using data_t = OrderedPoint<PointT>;
  using point_traits = cukd::point_traits<PointT>;
  using scalar_t = typename point_traits::scalar_t;

  static inline __device__ __host__ const PointT &
  get_point(const data_t &data) {
    return data.position;
  }

  static inline __device__ __host__ scalar_t get_coord(const data_t &data,
                                                       int dim) {
    return cukd::get_coord(get_point(data), dim);
  }

  enum { has_explicit_dim = false };
  static inline __device__ int get_dim(const data_t &) { return -1; }
};

// 复制内核
template <typename PointT>
__global__ void CopyKernel(OrderedPoint<PointT> *points, PointT *positions,
                           int n_batches, int n_points) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int nid = threadIdx.y + blockIdx.y * blockDim.y;

  if ((bid >= n_batches) || (nid >= n_points))
    return;

  // Row major
  int pid = bid * n_points + nid;
  points[pid].position = positions[pid];
  // Batch local index
  points[pid].idx = nid;
}

// 最近点查询内核
template <typename FloatT, typename PointT>
__global__ void
ClosestPointKernel(FloatT *d_dists, int *d_indices, PointT *d_queries,
                   int n_batches, int n_queries,
                   const cukd::box_t<PointT> *d_bounds,
                   OrderedPoint<PointT> *d_nodes, int n_points) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int mid = threadIdx.y + blockIdx.y * blockDim.y;

  if ((bid >= n_batches) || (mid >= n_queries))
    return;

  // Row major
  int qid = bid * n_queries + mid;
  PointT queryPos = d_queries[qid];
  cukd::FcpSearchParams params;
  // Local closest index
  int closestID =
      cukd::cct::fcp<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
          queryPos, *(d_bounds + bid), d_nodes + bid * n_points, n_points,
          params);
  int pid = bid * n_points + closestID;
  int idx = d_nodes[pid].idx;
  PointT inputPos = d_nodes[pid].position;

  float x_diff = queryPos.x - inputPos.x;
  float y_diff = queryPos.y - inputPos.y;
  float z_diff = queryPos.z - inputPos.z;

  d_dists[qid] = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
  d_indices[qid] = idx;
}

// CUDA函数：分配内存并构建KD树
template <typename FloatT, typename PointT, uint32_t THREAD_POOL,
          uint32_t BATCH_SIZE_B, uint32_t BATCH_SIZE_N, uint32_t BATCH_SIZE_M>
void buildKDTree(const torch::Tensor &input, void **d_nodes, void **d_bounds) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  uint32_t numBatches = input.size(0);
  uint32_t numInput = input.size(1);

  // We must copy because implicit tree will re-arange input data
  CUKD_CUDA_CHECK(cudaMallocAsync(
      d_nodes, numBatches * numInput * sizeof(OrderedPoint<PointT>), stream));

  // **IMPORTANT** We cannot loop, as data is in device memory
  CopyKernel<<<dim3(cukd::divRoundUp(numBatches, BATCH_SIZE_B),
                    cukd::divRoundUp(numInput, BATCH_SIZE_N)),
               dim3(BATCH_SIZE_B, BATCH_SIZE_N), 0, stream>>>(
      static_cast<OrderedPoint<PointT> *>(d_nodes),
      reinterpret_cast<PointT *>(input.data_ptr<FloatT>()), numBatches,
      numInput);
  cudaStreamSynchronize(stream);

  CUKD_CUDA_CHECK(cudaMallocAsync(
      d_bounds, numBatches * sizeof(cukd::box_t<PointT>), stream));

  // Build tree in parallel
  tcnn::SyncedMultiStream syncedStreams(stream, THREAD_POOL);
  omp_set_num_threads(THREAD_POOL);
#pragma omp parallel for schedule(dynamic)
  for (int bid = 0; bid < numBatches; bid++) {
    int tid = omp_get_thread_num();
    cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
        static_cast<OrderedPoint<PointT> *>(*d_nodes) + bid * numInput,
        numInput, static_cast<cukd::box_t<PointT> *>(*d_bounds) + bid,
        syncedStreams.get(tid));
    cudaStreamSynchronize(syncedStreams.get(tid));
  }
}

// CUDA函数：查询KD树
template <typename FloatT, typename PointT, uint32_t THREAD_POOL,
          uint32_t BATCH_SIZE_B, uint32_t BATCH_SIZE_N, uint32_t BATCH_SIZE_M>
std::vector<torch::Tensor> queryKDTree(void *d_nodes, void *d_bounds,
                                       const torch::Tensor &query,
                                       const uint32_t &n_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  uint32_t numBatches = query.size(0);
  uint32_t numQueries = query.size(0);

  const torch::TensorOptions distOpts =
      torch::TensorOptions().dtype(query.dtype()).device(query.device());
  torch::Tensor dists = torch::zeros({numBatches, numQueries}, distOpts);

  const torch::TensorOptions idxOpts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  torch::Tensor idxs = torch::zeros({numBatches, numQueries}, idxOpts);

  ClosestPointKernel<<<dim3(cukd::divRoundUp(numBatches, BATCH_SIZE_B),
                            cukd::divRoundUp(numQueries, BATCH_SIZE_M)),
                       dim3(BATCH_SIZE_B, BATCH_SIZE_M), 0, stream>>>(
      dists.data_ptr<FloatT>(), idxs.data_ptr<int>(),
      reinterpret_cast<PointT *>(query.data_ptr<FloatT>()), numBatches,
      numQueries, static_cast<cukd::box_t<PointT> *>(d_bounds),
      static_cast<OrderedPoint<PointT> *>(d_nodes), n_points);
  cudaStreamSynchronize(stream);

  return {dists, idxs};
}
