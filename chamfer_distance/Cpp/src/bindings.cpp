#include "chamfer_cpu.h"
#include "chamfer_cuda.h"
#include "chamfer_cuda_kd.h"
#include "chamfer_triton.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(chamfer_cpp, m) {
  m.doc() = "pybind11 chamfer cpp plugin";

  m.def("chamfer_cpu", &chamfer_cpu, "chamfer_cpu.chamfer_cpu");

  m.def("chamfer_cuda", &chamfer_cuda, "chamfer_cuda.chamfer_cuda");

  m.def("chamfer_cuda_kd", &chamfer_cuda_kd, "chamfer_cuda_kd.chamfer_cuda_kd");

  m.def("kd_closest_query_cuda", &kd_closest_query_cuda,
        "chamfer_triton.kd_closest_query_cuda");

  m.def("crude_nn_cuda", &crude_nn_cuda, "chamfer_triton.crude_nn_cuda");
}
