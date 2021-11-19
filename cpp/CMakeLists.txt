cmake_minimum_required(VERSION 3.19.0)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

#set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#et(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

project(cut C CXX)

option(BUILD_CUDA "Build with CUDA" ON)


enable_testing()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/gtest.cmake)

if (BUILD_CUDA)
  # NOTE: if you want to use source PTX with a version different from produced PTX/binary, you should add flags
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)
    # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
    list(APPEND CMAKE_CUDA_ARCHITECTURES 61-real)
    # V100, TITAN V
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      # A100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      # GeForce RTX 30xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-virtual)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual)
    else()
      list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)
    endif()
  endif()
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CUDA_SEPARABLE_COMPILATION OFF)

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
    set(CUDA_NVCC_THREADS_NUMBER "1" CACHE STRING "")
    list(APPEND CUDA_NVCC_FLAGS -t ${CUDA_NVCC_THREADS_NUMBER})
  endif()
  message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
  list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
endif()

message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

add_subdirectory(playground)
