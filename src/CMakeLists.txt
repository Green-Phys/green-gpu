project(gpu-lib CXX)
  
add_library(accel-lib cuda_common.cu cuda_check.cpp cublas_routines_prec.cu
        cu_compute_Pq.cu cugw_qpt.cu cu_routines.cu )
#see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
set(cuda_targets "70;72;80;86;90")
message("CUDA targets are: " ${cuda_targets})
set_property(TARGET accel-lib PROPERTY CUDA_ARCHITECTURES ${cuda_targets}) #also check the same parameters in solvers
target_include_directories(accel-lib PUBLIC .)
target_link_libraries(accel-lib CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
target_link_libraries(accel-lib GREEN::UTILS GREEN::NDARRAY)

add_library(gpu gpu_kernel.cpp hf_gpu_kernel.cpp gw_gpu_kernel.cpp)
target_include_directories(gpu PUBLIC .)
target_link_libraries(gpu CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
target_link_libraries(gpu accel-lib)
target_link_libraries(gpu GREEN::UTILS GREEN::NDARRAY GREEN::SYMMETRY GREEN::GRIDS GREEN::PARAMS)
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 70 72 80 86) #also check same parameters in accel
