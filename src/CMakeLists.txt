project(gpu-lib CXX)

find_package(MPI REQUIRED)
  
add_library(accel-lib cuda_common.cu cuda_check.cpp cublas_routines_prec.cu
        cu_compute_Pq.cu cugw_qpt.cu cu_routines.cu )
set_property(TARGET accel-lib PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS}) #also check the same parameters in solvers
target_include_directories(accel-lib PUBLIC .)
target_link_libraries(accel-lib CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
target_link_libraries(accel-lib GREEN::UTILS GREEN::NDARRAY GREEN::GRIDS GREEN::INTEGRALS GREEN::H5PP)

add_library(gpu gpu_kernel.cpp hf_gpu_kernel.cpp gw_gpu_kernel.cpp)
target_include_directories(gpu PUBLIC .)
target_link_libraries(gpu CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
target_link_libraries(gpu accel-lib)
target_link_libraries(gpu GREEN::UTILS GREEN::NDARRAY GREEN::SYMMETRY GREEN::GRIDS GREEN::PARAMS GREEN::INTEGRALS)
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS}) #also check same parameters in accel
