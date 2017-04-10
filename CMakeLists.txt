cmake_minimum_required(VERSION 3.5)
project(cuda-collatz)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/gpu_ptx_full)
set(UTIL ${CMAKE_SOURCE_DIR}/src/util)

find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)
#set(CUDA_SEPARABLE_COMPILATION ON)

find_package(Boost 1.58.0 EXACT REQUIRED COMPONENTS program_options system)

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
  message(WARNING "Building DEBUG version: ${CUDA_NVCC_FLAGS}")
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
  message(WARNING "Building RELEASE version: ${CUDA_NVCC_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SOURCE_DIR}
  ${UTIL}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)

link_libraries(
  ${Boost_LIBRARIES}
)

cuda_add_executable(
  collatz_ptx
  ${SOURCE_DIR}/collatz.cpp
  ${SOURCE_DIR}/collatz_kernel.cu
  ${UTIL}/cuda_util.cpp
)
