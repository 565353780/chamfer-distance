#include "chamfer_cuda.h"
#include "chamfer_cpu.h"

#ifdef USE_CUDA
std::vector<torch::Tensor>
chamfer_3DFunction::forward(torch::autograd::AutogradContext *ctx,
                            const torch::Tensor &xyz1,
                            const torch::Tensor &xyz2) {
  int batchsize = xyz1.size(0);
  int n = xyz1.size(1);
  int m = xyz2.size(1);

  torch::TensorOptions opts =
      torch::TensorOptions().dtype(xyz1.dtype()).device(xyz1.device());

  torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt).device(xyz1.device());

  torch::Tensor dist1 = torch::zeros({batchsize, n}, opts);
  torch::Tensor dist2 = torch::zeros({batchsize, m}, opts);

  torch::Tensor idx1 = torch::zeros({batchsize, n}, idx_opts);
  torch::Tensor idx2 = torch::zeros({batchsize, m}, idx_opts);

  chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);

  ctx->save_for_backward({xyz1, xyz2, idx1, idx2});

  std::vector<torch::Tensor> dists_with_idxs({dist1, dist2, idx1, idx2});

  return dists_with_idxs;
}

std::vector<torch::Tensor>
chamfer_3DFunction::backward(torch::autograd::AutogradContext *ctx,
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

const std::vector<torch::Tensor> chamfer_cuda(const torch::Tensor &input1,
                                              const torch::Tensor &input2) {
  const torch::Tensor contiguous_input1 = input1.contiguous();
  const torch::Tensor contiguous_input2 = input2.contiguous();

  const std::vector<torch::Tensor> dists_with_idxs =
      chamfer_3DFunction::apply(contiguous_input1, contiguous_input2);

  const torch::Tensor valid_dists1 = toValidTensor(dists_with_idxs[0]);
  const torch::Tensor valid_dists2 = toValidTensor(dists_with_idxs[1]);
  const torch::Tensor idxs1 = dists_with_idxs[2].toType(torch::kInt);
  const torch::Tensor idxs2 = dists_with_idxs[3].toType(torch::kInt);

  const std::vector<torch::Tensor> valid_dists_with_idxs(
      {valid_dists1, valid_dists2, idxs1, idxs2});

  return valid_dists_with_idxs;
}
#endif
