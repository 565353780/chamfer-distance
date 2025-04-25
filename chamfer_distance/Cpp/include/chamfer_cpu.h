#pragma once

#include <torch/extension.h>

const torch::Tensor toValidTensor(const torch::Tensor &source_tensor);

const torch::Tensor batched_pairwise_dist(const torch::Tensor &x,
                                          const torch::Tensor &y);

const std::vector<torch::Tensor> chamfer_cpu(const torch::Tensor &a,
                                             const torch::Tensor &b);
