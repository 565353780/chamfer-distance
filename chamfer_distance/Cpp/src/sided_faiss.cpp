#include "sided_faiss.h"
#include <ATen/cuda/CUDAContext.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#ifdef USE_CUDA
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
      reinterpret_cast<faiss::idx_t *>(idx1_flat.data_ptr<std::int64_t>()));
}
#endif
