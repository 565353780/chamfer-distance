#include "chamfer_cpu.h"

const torch::Tensor toValidTensor(const torch::Tensor &source_tensor) {
  const torch::Tensor valid_tensor = torch::nan_to_num(source_tensor, 0.0);
  return valid_tensor;
}

const torch::Tensor batched_pairwise_dist(const torch::Tensor &x,
                                          const torch::Tensor &y) {
  int bs = x.size(0);
  int num_points_x = x.size(1);
  int num_points_y = y.size(1);

  const torch::Tensor xx = torch::pow(x, 2).sum(2);
  const torch::Tensor yy = torch::pow(y, 2).sum(2);
  torch::Tensor zz;
  if (num_points_x < num_points_y) {
    zz = torch::bmm(2.0f * x, y.transpose(2, 1));
  } else {
    zz = torch::bmm(x, (2.0f * y).transpose(2, 1));
  }

  const torch::Tensor rx =
      xx.unsqueeze(2).expand({bs, num_points_x, num_points_y});
  const torch::Tensor ry =
      yy.unsqueeze(1).expand({bs, num_points_x, num_points_y});

  const torch::Tensor P = rx + ry - zz;

  return P;
}

const std::vector<torch::Tensor> chamfer_cpu(const torch::Tensor &a,
                                             const torch::Tensor &b) {
  const torch::Tensor P = batched_pairwise_dist(a, b);

  const std::tuple<torch::Tensor, torch::Tensor> P1 = torch::min(P, 2);
  const std::tuple<torch::Tensor, torch::Tensor> P2 = torch::min(P, 1);

  const torch::Tensor &dists1 = std::get<0>(P1);
  const torch::Tensor &dists2 = std::get<0>(P2);
  const torch::Tensor idxs1 = std::get<1>(P1).toType(torch::kInt);
  const torch::Tensor idxs2 = std::get<1>(P2).toType(torch::kInt);

  const torch::Tensor valid_dists1 = toValidTensor(dists1);
  const torch::Tensor valid_dists2 = toValidTensor(dists2);

  const std::vector<torch::Tensor> dists_with_idxs(
      {valid_dists1, valid_dists2, idxs1, idxs2});

  return dists_with_idxs;
}
