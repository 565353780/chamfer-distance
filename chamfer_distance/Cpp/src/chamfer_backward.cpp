#include "chamfer_backward.h"

int chamfer_torch_backward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                           const torch::Tensor &graddist1,
                           const torch::Tensor &graddist2,
                           const torch::Tensor &idx1, const torch::Tensor &idx2,
                           torch::Tensor &gradxyz1, torch::Tensor &gradxyz2) {
  const torch::Tensor valid_idxs1 =
      idx1.unsqueeze(-1).expand({-1, -1, 3}).toType(torch::kInt64);
  const torch::Tensor valid_idxs2 =
      idx2.unsqueeze(-1).expand({-1, -1, 3}).toType(torch::kInt64);

  const torch::Tensor d_dist1 = graddist1.unsqueeze(-1) * 2 *
                                (xyz1 - torch::gather(xyz2, 1, valid_idxs1));
  const torch::Tensor d_dist2 = graddist2.unsqueeze(-1) * 2 *
                                (xyz2 - torch::gather(xyz1, 1, valid_idxs2));

  gradxyz1 = torch::scatter_add(d_dist1, 1, valid_idxs2, -d_dist2);
  gradxyz2 = torch::scatter_add(d_dist2, 1, valid_idxs1, -d_dist1);

  return 1;
}
