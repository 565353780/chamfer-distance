#include "cukd_searcher.h"
#include "sided_cuda.h"
#include "sided_cukd.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(chamfer_cpp, m) {
  m.doc() = "pybind11 chamfer cpp plugin";

  m.def("sided_forward_cuda", &sided_forward_cuda,
        "sided_cuda.sided_forward_cuda");

  m.def("sided_forward_cukd", &sided_forward_cukd,
        "sided_cukd.sided_forward_cukd");

  // 绑定CUKDSearcher类
  pybind11::class_<CUKDSearcher>(m, "CUKDSearcher")
      .def(pybind11::init<>())
      .def("addPoints", &CUKDSearcher::addPoints, "添加点云数据并构建KD树")
      .def("query", &CUKDSearcher::query, "查询最近邻点");
}
