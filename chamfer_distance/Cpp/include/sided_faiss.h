#pragma once

#include <torch/extension.h>

#ifdef USE_CUDA
void sided_forward_faiss(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                         torch::Tensor &dist1, torch::Tensor &idx1);
#endif
