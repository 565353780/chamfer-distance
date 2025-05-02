#pragma once

#include <torch/extension.h>

#ifdef USE_CUDA
int chamfer_faiss_forward(const torch::Tensor &xyz1, // [B, N, 3]
                          const torch::Tensor &xyz2, // [B, M, 3]
                          torch::Tensor &dist1,      // [B, N]
                          torch::Tensor &dist2,      // [B, M]
                          torch::Tensor &idx1,       // [B, N]
                          torch::Tensor &idx2);      // [B, M]

class chamfer_FAISS_3DFunction
    : public torch::autograd::Function<chamfer_FAISS_3DFunction> {
public:
  static std::vector<torch::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &xyz1,
          const torch::Tensor &xyz2);

  static std::vector<torch::Tensor>
  backward(torch::autograd::AutogradContext *ctx,
           std::vector<torch::Tensor> &grad_outputs);
};

const std::vector<torch::Tensor> chamfer_faiss(const torch::Tensor &input1,
                                               const torch::Tensor &input2);
#endif
