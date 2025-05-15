#include "sided_faiss.h"
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
                reinterpret_cast<faiss::idx_t *>(idx.data_ptr<int64_t>()));

  // 返回距离和索引
  return {dist.squeeze(1), idx.squeeze(1)};
}

void sided_forward_faiss(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                         torch::Tensor &dist1, torch::Tensor &idx1) {
  TORCH_CHECK(xyz1.is_cuda() && xyz2.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(xyz1.dim() == 3 && xyz2.dim() == 3, "Input must be [B, N, 3]");

  const int B = xyz1.size(0);
  const int N = xyz1.size(1);
  const int M = xyz2.size(1);
  const int D = xyz1.size(2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 创建 FAISS 资源
  faiss::gpu::StandardGpuResources res;
  res.setDefaultStream(0, stream);

  // Flatten inputs
  auto xyz1_flat = xyz1.reshape({B * N, D}).contiguous();
  auto xyz2_flat = xyz2.reshape({B * M, D}).contiguous();

  auto dist1_flat = dist1.reshape({B * N}).contiguous();
  auto idx1_flat = idx1.reshape({B * N}).contiguous();

  // Build xyz2 as DB index
  faiss::gpu::GpuIndexFlatL2 index_x2(&res, D);
  index_x2.add(B * M, xyz2_flat.data_ptr<float>());

  // Query xyz1 against xyz2
  index_x2.search(
      B * N, xyz1_flat.data_ptr<float>(), 1, dist1_flat.data_ptr<float>(),
      reinterpret_cast<faiss::idx_t *>(idx1_flat.data_ptr<int64_t>()));
}
#endif
