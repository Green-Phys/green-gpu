cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(gpu)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

################ GREEN RELEASE VERSION #####################
set(GREEN_RELEASE origin/main)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if ( PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME )
    set(GIT_HASH "${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")
else ()
    set(GIT_HASH "${GIT_HASH}**${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")
endif ()

find_package(CUDAToolkit REQUIRED)
enable_language("CUDA")
include(CheckLanguage)
check_language(CUDA)
include(CudaArchitectures)
define_cuda_architectures()
add_compile_definitions(WITH_CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include(GreenKernel)
init_kernel("GPU" "GREEN::GPU" "<green/gpu/gpu_factory.h>" "green::gpu")

include(GreenDeps)

add_green_dependency(green-grids)
add_green_dependency(green-h5pp)
add_green_dependency(green-ndarray)
add_green_dependency(green-params)
add_green_dependency(green-symmetry)
add_green_dependency(green-utils)
add_green_dependency(green-integrals)

add_subdirectory(src)
add_library(GREEN::GPU ALIAS gpu)

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif ()
