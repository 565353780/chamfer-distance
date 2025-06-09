#include "sided_cukd.h"
#include "chamfer_triton.h"

#ifdef USE_CUDA
void sided_forward_cukd(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                        torch::Tensor &dist1, torch::Tensor &idx1) {
  // 检查输入张量的维度
  TORCH_CHECK(xyz1.dim() == 3, "xyz1 must be a 3D tensor of shape [B, N, 3]");
  TORCH_CHECK(xyz2.dim() == 3, "xyz2 must be a 3D tensor of shape [B, M, 3]");
  TORCH_CHECK(xyz1.size(0) == xyz2.size(0), "Batch sizes must match");
  
  std::vector<torch::Tensor> dist1_vec, idx1_vec;

  for (int i = 0; i < xyz1.size(0); ++i) {
    const std::vector<torch::Tensor> result1 =
        kd_closest_query_cuda(xyz1[i], xyz2[i]);

    dist1_vec.emplace_back(result1[0]);
    idx1_vec.emplace_back(result1[1]);
  }

  dist1.copy_(torch::vstack(dist1_vec));
  idx1.copy_(torch::vstack(idx1_vec));
}
#endif
