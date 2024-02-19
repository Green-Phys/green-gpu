/*
* Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once


#include <complex>
#include <cuComplex.h>

template<typename T>
struct cu_type_map {};

template<typename cuT>
struct get_type_map {};

template<>
struct get_type_map<cuComplex> {
  using type_map = cu_type_map<std::complex<float>>;
};

template<>
struct get_type_map<cuDoubleComplex> {
  using type_map = cu_type_map<std::complex<double>>;
};

template<>
struct cu_type_map<std::complex<double>> {

  using cxx_base_type = double;
  using cxx_type = std::complex<double>;
  using cuda_type = cuDoubleComplex;

  static __inline__ cuda_type cast(cxx_type in) {
    return make_cuDoubleComplex(in.real(), in.imag());
  }

  static __inline__ cuda_type cast(cxx_base_type in_r, cxx_base_type in_i) {
    return make_cuDoubleComplex(in_r, in_i);
  }

  __host__ __device__ static __inline__ cxx_base_type real(cuda_type x) {
    return cuCreal(x);
  }

  __host__ __device__ static __inline__ cxx_base_type imag(cuda_type x) {
    return cuCimag(x);
  }

  __host__ __device__ static __inline__ cuda_type add(cuda_type x, cuda_type y) {
    return cuCadd(x, y);
  }

  __host__ __device__ static __inline__ cuda_type mul(cuda_type x, cuda_type y) {
    return cuCmul(x, y);
  }
};

template<>
struct cu_type_map<std::complex<float>> {

  using cxx_base_type = float;
  using cxx_type = std::complex<float>;
  using cuda_type = cuComplex;

  static __inline__ cuda_type cast(cxx_type in) {
    return make_cuComplex(in.real(), in.imag());
  }

  static __inline__ cuda_type cast(cxx_base_type in_r, cxx_base_type in_i) {
    return make_cuComplex(in_r, in_i);
  }

  __host__ __device__ static __inline__ cxx_base_type real(cuda_type x) {
    return cuCrealf(x);
  }

  __host__ __device__ static __inline__ cxx_base_type imag(cuda_type x) {
    return cuCimagf(x);
  }

  __host__ __device__ static __inline__ cuda_type add(cuda_type x, cuda_type y) {
    return cuCaddf(x, y);
  }

  __host__ __device__ static __inline__ cuda_type mul(cuda_type x, cuda_type y) {
    return cuCmulf(x, y);
  }
};


