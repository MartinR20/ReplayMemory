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
    .def(py::init<const unsigned int, 
                  const unsigned int, 
                  const unsigned int,
                  const float,
                  const float,
                  const float,
                  const std::string>())
    .def("append", &ReplayMemory::append)
    .def("sample", &ReplayMemory::sample)
    .def("update_priorities", &ReplayMemory::update_priorities)
    .def("__iter__", &ReplayMemory::__iter__)
    .def("__next__", &ReplayMemory::__next__)
    .def_readwrite("priority_weight", &ReplayMemory::priority_weight)
  ;
}