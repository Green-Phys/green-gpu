function(define_cuda_architectures)
  #see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
  if (DEFINED GPU_ARCHS)
    message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    separate_arguments(GPU_ARCHS)
  else()
    list(APPEND GPU_ARCHS_
        70
        75
      )

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.0)
      # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
      list(APPEND GPU_ARCHS_ 80)
    endif()
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.1)
      list(APPEND GPU_ARCHS_ 86)
    endif()
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.1)
      list(APPEND GPU_ARCHS_ 90)
    endif()
    set(GPU_ARCHS ${GPU_ARCHS_} PARENT_SCOPE)
    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS_}")
  endif()


endfunction()