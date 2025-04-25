#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> kd_closest_query_cuda(const torch::Tensor &xyz1,
                                                 const torch::Tensor &xyz2);

torch::Tensor crude_nn_cuda(const torch::Tensor &xyz1,
                            const torch::Tensor &xyz2);
