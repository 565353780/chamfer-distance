#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include "crude_nn.cuh"
#include "kd_closest_query.cuh"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace triton_chamfer {

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

TORCH_LIBRARY(triton_chamfer, m) {
  m.def("kd_closest_query(Tensor a, Tensor b) -> Tensor[]");
  m.def("crude_nn(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(triton_chamfer, CUDA, m) {
  m.impl("kd_closest_query", &kd_closest_query_cuda);
  m.impl("crude_nn", &crude_nn_cuda);
}

} // namespace triton_chamfer
