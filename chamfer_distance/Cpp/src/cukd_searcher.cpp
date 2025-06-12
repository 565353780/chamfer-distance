#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>

CUKDSearcher::CUKDSearcher() : initialized(false) {}

CUKDSearcher::~CUKDSearcher() { releaseResources(); }

void CUKDSearcher::releaseResources() {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (d_nodes) {
    cudaFreeAsync(d_nodes, stream);
  }
  if (d_bounds) {
    cudaFreeAsync(d_bounds, stream);
  }

  batch_size = 0;
  n_points = 0;
}

// 添加点云数据并构建KD树
void CUKDSearcher::addPoints(const torch::Tensor &points) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, N, D]");
  TORCH_CHECK(points.size(2) == 3, "Input points must have 3 dimensions");

  releaseResources();

  buildKDTree(points, &d_nodes, &d_bounds);

  batch_size = points.size(0);
  n_points = points.size(1);

  initialized = true;
}

// 查询最近邻点
std::vector<torch::Tensor> CUKDSearcher::query(const torch::Tensor &points) {
  TORCH_CHECK(initialized, "No points have been added to the index yet");
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, M, D]");
  TORCH_CHECK(points.size(0) == batch_size,
              "Query batch size must match index batch size");
  TORCH_CHECK(points.size(2) == 3,
              "Query points dimension must match index dimension");

  // 调用CUDA函数执行查询
  return queryKDTree(d_nodes, d_bounds, points, n_points);
}
