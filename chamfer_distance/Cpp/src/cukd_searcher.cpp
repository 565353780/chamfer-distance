#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>

// 构造函数
CUKDSearcher::CUKDSearcher()
    : dimension(3), initialized(false), numInput(0), d_input(nullptr),
      d_bounds(nullptr) {}

// 析构函数
CUKDSearcher::~CUKDSearcher() { releaseResources(); }

// 释放资源
void CUKDSearcher::releaseResources() {
  if (d_input || d_bounds) {
    freeKDTreeResources(d_input, d_bounds);
    d_input = nullptr;
    d_bounds = nullptr;
  }
}

// 添加点云数据并构建KD树
void CUKDSearcher::addPoints(const torch::Tensor &points) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 2, "Input points must be [N, D]");
  TORCH_CHECK(points.size(1) == 3, "Input points must have 3 dimensions");

  // 释放之前的资源
  releaseResources();

  // 获取点云数量
  numInput = points.size(0);
  dimension = points.size(1);

  // 调用CUDA函数分配内存并构建KD树
  d_input = allocateAndBuildKDTree(points, &d_bounds);

  initialized = true;
}

// 查询最近邻点
std::vector<torch::Tensor> CUKDSearcher::query(const torch::Tensor &points) {
  TORCH_CHECK(initialized, "No points have been added to the index yet");
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 2, "Input points must be [M, D]");
  TORCH_CHECK(points.size(1) == dimension,
              "Query points dimension must match index dimension");

  // 调用CUDA函数执行查询
  return queryKDTree(d_input, d_bounds, points, numInput);
}

// 释放CUDA资源
void CUKDSearcher::freeKDTreeResources(void *d_input, void *d_bounds) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (d_input) {
    cudaFreeAsync(d_input, stream);
  }
  if (d_bounds) {
    cudaFreeAsync(d_bounds, stream);
  }
}
