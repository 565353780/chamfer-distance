#pragma once

#include <torch/extension.h>

#ifdef USE_CUDA
int chamfer_cuda_forward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                         torch::Tensor &dist1, torch::Tensor &dist2,
                         torch::Tensor &idx1, torch::Tensor &idx2);

class chamfer_3DFunction
    : public torch::autograd::Function<chamfer_3DFunction> {
public:
  static std::vector<torch::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &xyz1,
          const torch::Tensor &xyz2);

  static std::vector<torch::Tensor>
  backward(torch::autograd::AutogradContext *ctx,
           std::vector<torch::Tensor> &grad_outputs);
};

const std::vector<torch::Tensor> chamfer_cuda(const torch::Tensor &input1,
                                              const torch::Tensor &input2);
#endif
