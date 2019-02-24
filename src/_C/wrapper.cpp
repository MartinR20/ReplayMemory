#include "ReplayMemory.cpp"
#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  py::class_<AppendableSegmentTree<float>>(m, "SegmentTree")
    .def(py::init<const unsigned int>())
    .def("update", &AppendableSegmentTree<float>::update)
    .def("append", &AppendableSegmentTree<float>::append)
    .def("find", &AppendableSegmentTree<float>::find)
    .def("total", &AppendableSegmentTree<float>::total)
    .def("full", &AppendableSegmentTree<float>::full)
    .def("print", &AppendableSegmentTree<float>::print)
  ;

  py::class_<ReplayMemory>(m, "ReplayMemory")
    .def(py::init<const unsigned int, const unsigned int, const unsigned int>())
    .def("append", &ReplayMemory::append)
    .def("sample", &ReplayMemory::sample)
  ;
}