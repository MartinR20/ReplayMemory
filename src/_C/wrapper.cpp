#include "ReplayMemory.h"

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


  /*
  py::class_<Agent>(m, "Agent")
    .def(py::init<const unsigned int action_space,
                const unsigned int atoms,
                const float V_min,
                const float V_max,
                const at::Tensor support,
                const float delta_z,
                const unsigned int batch_size,
                const unsigned int multi_step;
                const float discount,
                torch::Module online_net,
                torch::Module target_net,
                const float lr,
                const unsigned int adam_eps,
                torch::Device device>())
    .def("reset_noise", &Agent::reset_noise)
    .def("act", &Agent::act)
    .def("act_e_greedy", &Agent::act_e_greedy)
    .def("learn", &Agent::learn)
    .def("update_target_net", &Agent::update_target_net)
    .def("save", &Agent::save)
    .def("evaluate_q", &Agent::evaluate_q)
    .def("train", &Agent::train)
    .def("eval", &Agent::eval)
  ;  */
}