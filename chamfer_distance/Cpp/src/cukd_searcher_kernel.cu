#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>
#include <cukd/builder.h>
#include <cukd/fcp.h>

// 定义OrderedPoint结构体
template <typename PointT> struct OrderedPoint {
  PointT position;
  int idx;
};

// 定义OrderedPoint_traits
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
                           int n_points) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_points)
    return;
  points[tid].position = positions[tid];
  points[tid].idx = tid;
}

// 最近点查询内核
template <typename T, typename PointT>
__global__ void
ClosestPointKernel(T *d_dists, int *d_indices, PointT *d_queries,
                   int numQueries, const cukd::box_t<PointT> *d_bounds,
                   OrderedPoint<PointT> *d_nodes, int numNodes) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;
  PointT queryPos = d_queries[tid];
  cukd::FcpSearchParams params;
  int closestID =
      cukd::cct::fcp<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
          queryPos, *d_bounds, d_nodes, numNodes, params);
  int idx = d_nodes[closestID].idx;
  PointT inputPos = d_nodes[closestID].position;

  float x_diff = queryPos.x - inputPos.x;
  float y_diff = queryPos.y - inputPos.y;
  float z_diff = queryPos.z - inputPos.z;
  d_dists[tid] = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;

  d_indices[tid] = idx;
}

// CUDA函数：分配内存并构建KD树
extern "C" void *allocateAndBuildKDTree(const torch::Tensor &points,
                                        void **d_bounds) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using PointT = float3;
  constexpr uint32_t BATCH_SIZE = 128;

  // 获取点云数量
  uint32_t numInput = points.size(0);

  // 分配内存
  void *d_input = nullptr;
  cudaMallocAsync(&d_input, numInput * sizeof(OrderedPoint<PointT>), stream);
  cudaMallocAsync(d_bounds, sizeof(cukd::box_t<PointT>), stream);

  // 复制数据
  CopyKernel<<<cukd::divRoundUp(numInput, BATCH_SIZE), BATCH_SIZE, 0, stream>>>(
      static_cast<OrderedPoint<PointT> *>(d_input),
      reinterpret_cast<PointT *>(points.data_ptr<float>()), numInput);

  // 构建KD树
  cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
      static_cast<OrderedPoint<PointT> *>(d_input), numInput,
      static_cast<cukd::box_t<PointT> *>(*d_bounds), stream);

  // 同步确保构建完成
  cudaStreamSynchronize(stream);

  return d_input;
}

// CUDA函数：查询KD树
extern "C" std::vector<torch::Tensor> queryKDTree(void *d_input, void *d_bounds,
                                                  const torch::Tensor &points,
                                                  int numInput) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using PointT = float3;
  using T = float;
  constexpr uint32_t BATCH_SIZE = 128;

  // 获取查询点数量
  uint32_t numQueries = points.size(0);

  // 创建输出张量
  const torch::TensorOptions distOpts =
      torch::TensorOptions().dtype(points.dtype()).device(points.device());
  torch::Tensor dists = torch::zeros({numQueries}, distOpts);

  const torch::TensorOptions idxOpts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());
  torch::Tensor idxs = torch::zeros({numQueries}, idxOpts);

  // 执行查询
  ClosestPointKernel<<<cukd::divRoundUp(numQueries, BATCH_SIZE), BATCH_SIZE, 0,
                       stream>>>(
      dists.data_ptr<T>(), idxs.data_ptr<int>(),
      reinterpret_cast<PointT *>(points.data_ptr<T>()), numQueries,
      static_cast<cukd::box_t<PointT> *>(d_bounds),
      static_cast<OrderedPoint<PointT> *>(d_input), numInput);

  // 同步确保查询完成
  cudaStreamSynchronize(stream);

  return {dists, idxs};
}

// CUDA函数：释放KD树资源
extern "C" void freeKDTreeResources(void *d_input, void *d_bounds) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (d_input) {
    cudaFreeAsync(d_input, stream);
  }
  if (d_bounds) {
    cudaFreeAsync(d_bounds, stream);
  }
}
