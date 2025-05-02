#pragma once

#include <torch/extension.h>

int chamfer_torch_backward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                           const torch::Tensor &graddist1,
                           const torch::Tensor &graddist2,
                           const torch::Tensor &idx1, const torch::Tensor &idx2,
                           torch::Tensor &gradxyz1, torch::Tensor &gradxyz2);

#ifdef USE_CUDA
int chamfer_cuda_backward(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                          const torch::Tensor &graddist1,
                          const torch::Tensor &graddist2,
                          const torch::Tensor &idx1, const torch::Tensor &idx2,
                          torch::Tensor &gradxyz1, torch::Tensor &gradxyz2);
#endif
