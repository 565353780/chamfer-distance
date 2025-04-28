#include "chamfer_cuda_kd_cub.h"
#include "chamfer_cpu.h"
#include "chamfer_cuda.h"
#include "chamfer_triton.h"

#ifdef USE_CUDA
std::vector<torch::Tensor>
chamfer_KDCUB_3DFunction::forward(torch::autograd::AutogradContext *ctx,
                                  const torch::Tensor &xyz1,
                                  const torch::Tensor &xyz2) {
  std::vector<torch::Tensor> dist1_vec, dist2_vec, idx1_vec, idx2_vec;

  for (int i = 0; i < xyz1.size(0); ++i) {
    std::vector<torch::Tensor> result1 =
        kd_closest_query_cuda(xyz1[i], xyz2[i]);
    std::vector<torch::Tensor> result2 =
        kd_closest_query_cuda(xyz2[i], xyz1[i]);

    const torch::Tensor &dist1 = result1[0];
    const torch::Tensor &idx1 = result1[1];
    const torch::Tensor &dist2 = result2[0];
    const torch::Tensor &idx2 = result2[1];

    dist1_vec.emplace_back(dist1);
    dist2_vec.emplace_back(dist2);
    idx1_vec.emplace_back(idx1);
    idx2_vec.emplace_back(idx2);
  }

  const torch::Tensor dists1 = torch::vstack(dist1_vec);
  const torch::Tensor dists2 = torch::vstack(dist2_vec);
  const torch::Tensor idxs1 = torch::vstack(idx1_vec);
  const torch::Tensor idxs2 = torch::vstack(idx2_vec);

  ctx->save_for_backward({xyz1, xyz2, idxs1, idxs2});

  std::vector<torch::Tensor> dists_with_idxs({dists1, dists2, idxs1, idxs2});

  return dists_with_idxs;
}

std::vector<torch::Tensor>
chamfer_KDCUB_3DFunction::backward(torch::autograd::AutogradContext *ctx,
                                   std::vector<torch::Tensor> &grad_outputs) {
  std::vector<torch::Tensor> dists_with_idxs = ctx->get_saved_variables();

  torch::Tensor xyz1 = dists_with_idxs[0];
  torch::Tensor xyz2 = dists_with_idxs[1];
  torch::Tensor idx1 = dists_with_idxs[2];
  torch::Tensor idx2 = dists_with_idxs[3];

  torch::Tensor &graddist1 = grad_outputs[0];
  torch::Tensor &graddist2 = grad_outputs[1];

  torch::Tensor contiguous_graddist1 = graddist1.contiguous();
  torch::Tensor contiguous_graddist2 = graddist2.contiguous();

  torch::Tensor gradxyz1 = torch::zeros_like(xyz1);
  torch::Tensor gradxyz2 = torch::zeros_like(xyz2);

  chamfer_cuda_backward(xyz1, xyz2, graddist1, graddist2, idx1, idx2, gradxyz1,
                        gradxyz2);

  std::vector<torch::Tensor> grads({gradxyz1, gradxyz2});

  return grads;
}

const std::vector<torch::Tensor>
chamfer_cuda_kd_cub(const torch::Tensor &input1, const torch::Tensor &input2) {
  const torch::Tensor contiguous_input1 = input1.contiguous();
  const torch::Tensor contiguous_input2 = input2.contiguous();

  const std::vector<torch::Tensor> dists_with_idxs =
      chamfer_KDCUB_3DFunction::apply(contiguous_input1, contiguous_input2);

  const torch::Tensor valid_dists1 = toValidTensor(dists_with_idxs[0]);
  const torch::Tensor valid_dists2 = toValidTensor(dists_with_idxs[1]);
  const torch::Tensor idxs1 = dists_with_idxs[2].toType(torch::kInt);
  const torch::Tensor idxs2 = dists_with_idxs[3].toType(torch::kInt);

  const std::vector<torch::Tensor> valid_dists_with_idxs(
      {valid_dists1, valid_dists2, idxs1, idxs2});

  return valid_dists_with_idxs;
}
#endif
