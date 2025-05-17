#include "faiss_searcher.h"
#include "sided_cuda.h"
#include "sided_cukd.h"
#include "sided_faiss.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(chamfer_cpp, m) {
  m.doc() = "pybind11 chamfer cpp plugin";

  m.def("sided_forward_cuda", &sided_forward_cuda,
        "sided_cuda.sided_forward_cuda");

  m.def("sided_forward_cukd", &sided_forward_cukd,
        "sided_cukd.sided_forward_cukd");

  m.def("sided_forward_faiss", &sided_forward_faiss,
        "sided_faiss.sided_forward_faiss");

  // 绑定FAISSSearcher类
  pybind11::class_<FAISSSearcher>(m, "FAISSSearcher")
      .def(pybind11::init<>())
      .def("addPoints", &FAISSSearcher::addPoints, "添加点云数据到索引")
      .def("query", &FAISSSearcher::query, "查询最近邻点");
}
