#include "faiss_searcher.h"
#include <ATen/cuda/CUDAContext.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#ifdef USE_CUDA
// FAISSSearcher实现
FAISSSearcher::FAISSSearcher() : dimension(0), initialized(false) {
  res = std::make_unique<faiss::gpu::StandardGpuResources>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  res->setDefaultStream(0, stream);
}

FAISSSearcher::~FAISSSearcher() {}

void FAISSSearcher::addPoints(const torch::Tensor &points) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 2, "Input points must be [N, D]");

  const int N = points.size(0);
  const int D = points.size(1);

  // 如果是第一次添加点，初始化索引
  if (!initialized || dimension != D) {
    dimension = D;
    index = std::make_unique<faiss::gpu::GpuIndexFlatL2>(res.get(), D);
    initialized = true;
  }

  // 确保点云数据是连续的
  auto points_contiguous = points.contiguous();

  // 添加点云到索引
  index->add(N, points_contiguous.data_ptr<float>());
}

std::vector<torch::Tensor> FAISSSearcher::query(const torch::Tensor &points) {
  TORCH_CHECK(points.is_cuda(), "Input points must be CUDA tensor");
  TORCH_CHECK(points.dim() == 2, "Input points must be [M, D]");
  TORCH_CHECK(initialized, "No points have been added to the index yet");
  TORCH_CHECK(points.size(1) == dimension,
              "Query points dimension must match index dimension");

  const int M = points.size(0);

  // 创建输出张量
  torch::TensorOptions opts =
      torch::TensorOptions().dtype(points.dtype()).device(points.device());
  torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(points.device());

  torch::Tensor dist = torch::zeros({M, 1}, opts);
  torch::Tensor idx = torch::zeros({M, 1}, idx_opts);

  // 确保查询点是连续的
  auto points_contiguous = points.contiguous();

  // 执行查询
  index->search(M, points_contiguous.data_ptr<float>(), 1,
                dist.data_ptr<float>(),
                reinterpret_cast<faiss::idx_t *>(idx.data_ptr<std::int64_t>()));

  // 返回距离和索引
  return {dist.squeeze(1), idx.squeeze(1)};
}
#endif
