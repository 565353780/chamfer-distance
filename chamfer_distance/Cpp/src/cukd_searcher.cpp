#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>

// 构造函数
CUKDSearcher::CUKDSearcher() : dimension(3), initialized(false), batchSize(0) {}

// 析构函数
CUKDSearcher::~CUKDSearcher() { releaseResources(); }

// 释放资源
void CUKDSearcher::releaseResources() {
  for (int i = 0; i < batchSize; ++i) {
    if (d_inputs[i] || d_bounds[i]) {
      freeKDTreeResources(d_inputs[i], d_bounds[i]);
    }
  }
  d_inputs.clear();
  d_bounds.clear();
  numInputs.clear();
  batchSize = 0;
}

// 添加点云数据并构建KD树
void CUKDSearcher::addPoints(const torch::Tensor &points) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, N, D]");
  TORCH_CHECK(points.size(2) == 3, "Input points must have 3 dimensions");

  // 释放之前的资源
  releaseResources();

  // 获取批次大小和点云数量
  batchSize = points.size(0);
  dimension = points.size(2);

  // 调整容器大小
  d_inputs.resize(batchSize, nullptr);
  d_bounds.resize(batchSize, nullptr);
  numInputs.resize(batchSize, 0);

  // 为每个批次构建KD树
  for (int i = 0; i < batchSize; ++i) {
    numInputs[i] = points.size(1);
    d_inputs[i] = allocateAndBuildKDTree(points[i], &d_bounds[i], i);
  }

  initialized = true;
}

// 查询最近邻点
std::vector<torch::Tensor> CUKDSearcher::query(const torch::Tensor &points) {
  TORCH_CHECK(initialized, "No points have been added to the index yet");
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, M, D]");
  TORCH_CHECK(points.size(0) == batchSize,
              "Query batch size must match index batch size");
  TORCH_CHECK(points.size(2) == dimension,
              "Query points dimension must match index dimension");

  // 调用CUDA函数执行查询
  return queryKDTree(d_inputs, d_bounds, points, numInputs);
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
