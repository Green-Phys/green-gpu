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

//
/**
 * \brief allocate G and Sigma on the device, and copy G from host to device
 * \param G pointer to Green's function array
 * \param Sigma pointer to self-energy array
 * \param G_host pointer to a host-stored Green's function array
 * \param nk - number of k points
 * \param nao number of orbitals
 * \param nt number of imaginary time points
 */
void allocate_G_and_Sigma(cuDoubleComplex ** G, cuDoubleComplex **Sigma, std::complex<double> *G_host, int nk, int nao, int nt);

//same, but also allocate and copy -G given G
//
/**
 * \brief allocate G(tau), G(-tau) and Sigma(tau) on the device, and copy G from host to device
 *
 * \tparam cu_type type of cuda scalar (single or double complex)
 * \param G_kstij pointer to Green's function array for positive  imaginary time
 * \param G_ksmtij pointer to Green's function array for negative imaginary time
 * \param Sigma_kstij pointer to self-energy array
 * \param G_tskij_host pointer to a host-stored Green's function array
 * \param nk - number of k points
 * \param nao number of orbitals
 * \param nt number of imaginary time points
 */
template<typename cu_type>
void allocate_G_and_Sigma(cu_type ** G_kstij, cu_type **G_ksmtij, cu_type **Sigma_kstij,
                          const std::complex<double> *G_tskij_host, int nk, int nao, int nt, int ns);

/**
 * \brief Allocate self-energy in device memory
 * \tparam cu_type type of cuda scalar (single or double complex)
 * \param Sigma_kstij pointer to a self-energy array
 * \param nk number of k-points
 * \param nao number of orbitals
 * \param nt number of imaginary time points
 * \param ns number of spins
 */
template<typename cu_type>
void allocate_Sigma(cu_type **Sigma_kstij, int nk, int nao, int nt, int ns);

/**
 * \brief allocate density and Fock matrix in the device memory and copy denisty matrix from host
 * \param Dm_fbz_skij_device denisty-matrix array
 * \param F_skij_device fock-matrix array
 * \param Dm_fbz_skij_host density matrix array stored in host memry
 * \param ink number of k-points in reduced Brillouin zone
 * \param nk number of k-points
 * \param nao number of orbitals
 * \param ns number of spins
 */
void allocate_density_and_Fock(cuDoubleComplex** Dm_fbz_skij_device, cuDoubleComplex** F_skij_device,
        const std::complex<double> *Dm_fbz_skij_host, int ink, int nk, int nao, int ns);
/**
 * \brief Allocate and copy k-point weights
 * \param weights_fbz_device k-point weight array
 * \param weights_fbz_host k-point weight array
 * \param ink number of k-points in the reduced Brillouin zone
 * \param nk number of k-points in the full Brillouin zone
 */
void allocate_weights(cuDoubleComplex** weights_fbz_device, const std::complex<double> *weights_fbz_host, int ink, int nk);

/**
 * \brief Allocate and copy transformation matrices for time-frequency transformations
 * \tparam cu_type type of cuda scalar (single or double complex)
 * \param T_tw_fb frequency to time transformation matrix
 * \param T_wt_bf time to fequency transformation matrix
 * \param T_tw_fb_host frequency to time transformation matrix (host)
 * \param T_wt_bf_host time to fequency transformation matrix (host)
 * \param nt number of imaginary time points
 * \param nw_b number of Matsubara frequency points
 */
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

