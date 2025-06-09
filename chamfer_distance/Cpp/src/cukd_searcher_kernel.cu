#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>
#include <cukd/builder.h>
#include <cukd/fcp.h>

// 定义OrderedPoint_traits
template <typename T>
struct OrderedPoint_traits : public cukd::default_data_traits<T> {
  using data_t = OrderedPoint<T>;
  using point_traits = cukd::point_traits<T>;
  using scalar_t = typename point_traits::scalar_t;

  static inline __device__ __host__ const T &get_point(const data_t &data) {
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
template <typename T>
__global__ void CopyKernel(OrderedPoint<T> *points, T *positions,
                           int n_points) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_points)
    return;
  points[tid].position = positions[tid];
  points[tid].idx = tid;
}

// 最近点查询内核
template <typename FloatT, typename T>
__global__ void ClosestPointKernel(FloatT *d_dists, int *d_indices,
                                   T *d_queries, int numQueries,
                                   const cukd::box_t<T> *d_bounds,
                                   OrderedPoint<T> *d_nodes, int numNodes) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;

  T queryPos = d_queries[tid];
  cukd::FcpSearchParams params;

  int closestID = cukd::cct::fcp<OrderedPoint<T>, OrderedPoint_traits<T>>(
      queryPos, *d_bounds, d_nodes, numNodes, params);

  // 确保找到有效的最近邻
  if (closestID >= 0 && closestID < numNodes) {
    int idx = d_nodes[closestID].idx;
    T inputPos = d_nodes[closestID].position;

    float x_diff = queryPos.x - inputPos.x;
    float y_diff = queryPos.y - inputPos.y;
    float z_diff = queryPos.z - inputPos.z;
    d_dists[tid] = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
    d_indices[tid] = idx;
  } else {
    // 如果没有找到有效的最近邻，设置一个无效值
    d_dists[tid] = INFINITY;
    d_indices[tid] = -1;
  }
}

// CUDA函数：分配内存并构建KD树
void *CUKDSearcher::allocateAndBuildKDTree(const torch::Tensor &points,
                                           void **d_bounds, int batchIdx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr uint32_t KERNEL_BATCH_SIZE = 256; // 增加批处理大小以提高性能

  // 获取点云数量
  uint32_t numInput = points.size(0);
  if (numInput == 0) {
    return nullptr;
  }

  // 分配内存
  void *d_input = nullptr;
  cudaError_t err;

  err = cudaMallocAsync(&d_input, numInput * sizeof(OrderedPoint<PointT>),
                        stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate memory for input points");
  }

  err = cudaMallocAsync(d_bounds, sizeof(cukd::box_t<PointT>), stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_input, stream);
    throw std::runtime_error("Failed to allocate memory for bounds");
  }

  // 复制数据
  dim3 block(KERNEL_BATCH_SIZE);
  dim3 grid(cukd::divRoundUp(numInput, KERNEL_BATCH_SIZE));

  CopyKernel<<<grid, block, 0, stream>>>(
      static_cast<OrderedPoint<PointT> *>(d_input),
      reinterpret_cast<PointT *>(points.data_ptr<float>()), numInput);

  // 检查kernel执行错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFreeAsync(d_input, stream);
    cudaFreeAsync(*d_bounds, stream);
    throw std::runtime_error("Failed to execute copy kernel");
  }

  // 构建KD树
  try {
    cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
        static_cast<OrderedPoint<PointT> *>(d_input), numInput,
        static_cast<cukd::box_t<PointT> *>(*d_bounds), stream);
  } catch (const std::exception &e) {
    cudaFreeAsync(d_input, stream);
    cudaFreeAsync(*d_bounds, stream);
    throw;
  }

  // 同步确保构建完成
  cudaStreamSynchronize(stream);

  return d_input;
}

// CUDA函数：查询KD树
std::vector<torch::Tensor> CUKDSearcher::queryKDTree(
    const std::vector<void *> &d_inputs, const std::vector<void *> &d_bounds,
    const torch::Tensor &points, const std::vector<int> &numInputs) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using FloatT = float;
  constexpr uint32_t KERNEL_BATCH_SIZE = 256; // 增加批处理大小以提高性能

  // 获取批次大小和每批次的查询点数量
  int batchSize = points.size(0);
  uint32_t numQueries = points.size(1);

  if (numQueries == 0) {
    return {
        torch::empty({batchSize, 0}, points.options()),
        torch::empty({batchSize, 0}, points.options().dtype(torch::kInt32))};
  }

  // 创建输出张量
  const torch::TensorOptions distOpts =
      torch::TensorOptions().dtype(points.dtype()).device(points.device());
  torch::Tensor dists = torch::zeros({batchSize, numQueries}, distOpts);

  const torch::TensorOptions idxOpts =
      torch::TensorOptions().dtype(torch::kInt32).device(points.device());
  torch::Tensor idxs = torch::zeros({batchSize, numQueries}, idxOpts);

  // 为每个批次执行查询
  dim3 block(KERNEL_BATCH_SIZE);
  dim3 grid(cukd::divRoundUp(numQueries, KERNEL_BATCH_SIZE));

  for (int i = 0; i < batchSize; ++i) {
    if (!d_inputs[i] || !d_bounds[i] || numInputs[i] == 0) {
      continue;
    }

    // 执行查询
    ClosestPointKernel<<<grid, block, 0, stream>>>(
        dists.index({i}).data_ptr<FloatT>(), idxs.index({i}).data_ptr<int>(),
        reinterpret_cast<PointT *>(points.index({i}).data_ptr<FloatT>()),
        numQueries, static_cast<cukd::box_t<PointT> *>(d_bounds[i]),
        static_cast<OrderedPoint<PointT> *>(d_inputs[i]), numInputs[i]);

    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to execute query kernel");
    }
  }

  // 同步确保查询完成
  cudaStreamSynchronize(stream);

  return {dists, idxs};
}
