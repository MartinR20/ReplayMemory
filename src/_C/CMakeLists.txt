cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(component_test)

#set(CMAKE_CXX_CLANG_TIDY "/usr/bin/clang-tidy;-checks=bugprone-, cert-, cppcoreguidelines-, clang-analyzer-, google-, hicpp-, llvm-, misc-, modernize-, mpi-, performance-, portability-, readability-")
#set(CMAKE_CXX_CPPCHECK "/usr/bin/cppcheck;--std=c++11")
#set(CMAKE_CXX_CPPLINT "/home/notroot/.local/bin/cpplint;--linelength=79")
#set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "/usr/bin/iwyu;--transitive_includes_only")
#set(CMAKE_LINK_WHAT_YOU_USE TRUE)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

set(CMAKE_PREFIX_PATH "~/Projects/ml/rl/torch/libtorch")

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

add_executable(component_test benchmark.cpp)

include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(component_test ${PYTHON_LIBRARIES})
target_link_libraries(component_test ${TORCH_LIBRARIES})