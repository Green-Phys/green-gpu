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

#include <stdexcept>
#include "cuComplex.h"
#include "cuda_types_map.h"

//allocate G and Sigma on the device, and copy G from host to device
void allocate_G_and_Sigma(cuDoubleComplex ** G, cuDoubleComplex **Sigma, std::complex<double> *G_host, int nk, int nao, int nt);

//same, but also allocate and copy -G given G
template<typename cu_type>
void allocate_G_and_Sigma(cu_type ** G_kstij, cu_type **G_ksmtij, cu_type **Sigma_kstij,
                          const std::complex<double> *G_tskij_host, int nk, int nao, int nt, int ns);

//void allocate_Sigma(cuDoubleComplex **Sigma_kstij, int nk, int nao, int nt, int ns);
template<typename cu_type>
void allocate_Sigma(cu_type **Sigma_kstij, int nk, int nao, int nt, int ns);

void allocate_density_and_Fock(cuDoubleComplex** Dm_fbz_skij_device, cuDoubleComplex** F_skij_device,
        const std::complex<double> *Dm_fbz_skij_host, int nk, int fnk, int nao, int ns);
void allocate_weights(cuDoubleComplex** weights_fbz_device, const std::complex<double> *weights_fbz_host, int ink, int nk);

// allocate IR transformation matrices T_tw_fb and T_wt_bf
template<typename cu_type>
void allocate_IR_transformation_matrices(cu_type ** T_tw_fb, cu_type ** T_wt_bf,
                                         const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host, int nt, int nw_b);

void Complex_DoubleToFloat(const std::complex<double>* in, std::complex<float>* out, size_t size);
void Complex_FloatToDouble(const std::complex<float>* in, std::complex<double>* out, size_t size);

//copy Sigma from the device to the host after calculation is done.
template<typename cu_type>
void copy_Sigma_from_device_to_host(cu_type *sigma_kstij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt, int ns);

template<typename cu_type>
void copy_2c_Sigma_from_device_to_host(cu_type *sigma_k4tij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt);

void copy_2c_Fock_from_device_to_host(cuDoubleComplex *F_3kij_device, std::complex<double> *F_kij_host, int nk, int nao);

void copy_Fock_from_device_to_host(cuDoubleComplex *F_skij_device, std::complex<double> *F_host, int nk, int nao, int ns);

__global__ void acquire_lock(int *lock);
__global__ void release_lock(int *lock);
__global__ void set_batch_pointer(cuDoubleComplex** ptrs_to_set, cuDoubleComplex *start, int stride, int N);
__global__ void set_batch_pointer(cuComplex** ptrs_to_set, cuComplex *start, int stride, int N);
__global__ void hermitian_symmetrize(cuDoubleComplex *M, int size);
__global__ void hermitian_symmetrize(cuComplex *M, int size);

