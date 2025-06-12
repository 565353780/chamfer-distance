#include "sided_cukd.h"
#include "cukd_searcher.h"

#ifdef USE_CUDA
void sided_forward_cukd(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                        torch::Tensor &dist1, torch::Tensor &idx1) {
  // 检查输入张量的维度
  TORCH_CHECK(xyz1.dim() == 3, "xyz1 must be a 3D tensor of shape [B, N, 3]");
  TORCH_CHECK(xyz2.dim() == 3, "xyz2 must be a 3D tensor of shape [B, M, 3]");
  TORCH_CHECK(xyz1.size(0) == xyz2.size(0), "Batch sizes must match");

  std::cout << "create searcher class\n";
  CUKDSearcher searcher;

  std::cout << "addPoints\n";
  searcher.addPoints(xyz2);

  std::cout << "query\n";
  std::vector<torch::Tensor> results = searcher.query(xyz1);

  std::cout << "copy results\n";
  dist1.copy_(results[0]);
  idx1.copy_(results[1]);
}
#endif
