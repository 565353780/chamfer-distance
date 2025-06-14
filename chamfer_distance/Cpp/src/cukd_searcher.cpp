#include "cukd_searcher.h"
#include <ATen/cuda/CUDAContext.h>

CUKDSearcher::CUKDSearcher() {}

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

  initialized = false;
}

void CUKDSearcher::addPoints(const torch::Tensor &points,
                             const uint32_t &THREAD_POOL,
                             const uint32_t &BATCH_SIZE_B,
                             const uint32_t &BATCH_SIZE_N) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, N, D]");
  TORCH_CHECK(points.size(2) == 3, "Input points must have 3 dimensions");

  releaseResources();

  buildKDTree<float, float3>(points, &d_nodes, &d_bounds, THREAD_POOL,
                             BATCH_SIZE_B, BATCH_SIZE_N);

  batch_size = points.size(0);
  n_points = points.size(1);

  initialized = true;
}

std::vector<torch::Tensor> CUKDSearcher::query(const torch::Tensor &points,
                                               const uint32_t &BATCH_SIZE_B,
                                               const uint32_t &BATCH_SIZE_N) {
  TORCH_CHECK(initialized, "No points have been added to the index yet");
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Input points must be [B, M, D]");
  TORCH_CHECK(points.size(0) == batch_size,
              "Query batch size must match index batch size");
  TORCH_CHECK(points.size(2) == 3,
              "Query points dimension must match index dimension");

  int numBatches = points.size(0);
  int numQueries = points.size(1);

  auto device = points.device();
  auto dtype = points.dtype();

  torch::Tensor dists = torch::zeros({numBatches, numQueries},
                                     torch::dtype(dtype).device(device));

  torch::Tensor idxs = torch::zeros({numBatches, numQueries},
                                    torch::dtype(torch::kInt32).device(device));

  queryKDTree<float, float3>(d_nodes, d_bounds, points, n_points, dists, idxs,
                             BATCH_SIZE_B, BATCH_SIZE_N);

  return {dists, idxs};
}
