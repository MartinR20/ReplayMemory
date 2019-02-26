cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(component_test)

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

add_executable(component_test ReplayMemory.cpp SegmentTree.cpp component_test.cpp)

include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(component_test ${PYTHON_LIBRARIES})
target_link_libraries(component_test ${TORCH_LIBRARIES})