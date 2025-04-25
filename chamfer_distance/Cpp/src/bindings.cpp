#include "module.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(chamfer_cpp, m) {
  m.doc() = "pybind11 chamfer cpp plugin";

  m.def("kd_closest_query_cuda", &kd_closest_query_cuda,
        "module.kd_closest_query_cuda");

  m.def("crude_nn_cuda", &crude_nn_cuda, "module.crude_nn_cuda");

  // m.def("toChamferDistance", &toChamferDistance,
  // "chamfer.toChamferDistance");
}
