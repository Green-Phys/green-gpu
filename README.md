[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-mbpt?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![gpu](https://github.com/Green-Phys/green-gpu/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Green-Phys/green-gpu/graph/badge.svg?token=GZGIYJ52PW)](https://codecov.io/gh/Green-Phys/green-gpu)

# green-gpu
Implementation of HF/GW kernels for GPU

Set the following cmake variables for custom kernel extension in green-mpt project
   - GREEN_KERNEL_URL="https://github.com/Green-Phys/green-gpu" 
   - GREEN_CUSTOM_KERNEL_LIB="GREEN::GPU"
   - GREEN_CUSTOM_KERNEL_ENUM=GPU 
   - GREEN_CUSTOM_KERNEL_HEADER=\<green/gpu/gpu_factory.h\>

# Acknowledgements

This work is supported by National Science Foundation under the award CSSI-2310582
