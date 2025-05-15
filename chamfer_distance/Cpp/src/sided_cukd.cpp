#include "sided_cukd.h"
#include "chamfer_triton.h"

#ifdef USE_CUDA
void sided_forward_cukd(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                        torch::Tensor &dist1, torch::Tensor &idx1) {
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
