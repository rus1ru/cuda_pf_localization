cmake_minimum_required(VERSION 3.18)
project(cuda_pf_localization_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(pf_kernels SHARED src/cuda/pf_kernels.cu)
target_include_directories(pf_kernels PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(pf_kernels PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET default
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build)
