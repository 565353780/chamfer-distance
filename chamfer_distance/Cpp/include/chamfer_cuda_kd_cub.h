#pragma once

#include <torch/extension.h>

#ifdef USE_CUDA
class chamfer_KDCUB_3DFunction
    : public torch::autograd::Function<chamfer_KDCUB_3DFunction> {
public:
  static std::vector<torch::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &xyz1,
          const torch::Tensor &xyz2);

  static std::vector<torch::Tensor>
  backward(torch::autograd::AutogradContext *ctx,
           std::vector<torch::Tensor> &grad_outputs);
};

const std::vector<torch::Tensor>
chamfer_cuda_kd_cub(const torch::Tensor &input1, const torch::Tensor &input2);
#endif
