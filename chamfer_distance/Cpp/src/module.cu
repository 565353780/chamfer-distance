
#include "crude_nn.cuh"
#include "kd_closest_query.cuh"
#include "module.h"
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

std::vector<torch::Tensor> kd_closest_query_cuda(const torch::Tensor &xyz1,
                                                 const torch::Tensor &xyz2) {
  TORCH_CHECK(xyz1.size(1) == xyz2.size(1));
  TORCH_INTERNAL_ASSERT(xyz1.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(xyz2.device().type() == at::DeviceType::CUDA);

  const at::cuda::CUDAGuard device_guard{xyz1.device()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return KDQueryClosest(stream, xyz2.contiguous(), xyz1.contiguous());
}

torch::Tensor crude_nn_cuda(const torch::Tensor &xyz1,
                            const torch::Tensor &xyz2) {
  TORCH_CHECK(xyz1.size(1) == xyz2.size(1));
  TORCH_INTERNAL_ASSERT(xyz1.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(xyz2.device().type() == at::DeviceType::CUDA);

  const at::cuda::CUDAGuard device_guard{xyz1.device()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return CrudeNN(stream, xyz2.contiguous(), xyz1.contiguous());
}
