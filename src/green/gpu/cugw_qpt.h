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
#include <cusolverDn.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common_defs.h"
#include "cublas_routines_prec.h"
#include "cuda_common.h"

__global__ void validate_info(int* info);
__global__ void validate_info(int* info, int N);
__global__ void set_up_one_minus_P(cuDoubleComplex* one_minus_P, cuDoubleComplex* P, int naux);
__global__ void set_up_one_minus_P(cuComplex* one_minus_P, cuComplex* P, int naux);

namespace green::gpu {

  /**
   * \brief cuda worker for quantities that have only q momentum dependence.
   *
   * \tparam prec single or double precision
   */
  template <typename prec>
  class gw_qpt {
    using scalar_t     = typename cu_type_map<std::complex<prec>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;

  public:
    gw_qpt(int nao, int naux, int ns, int nt, int nw_b, const std::complex<double>* T_tw_fb_host,
           const std::complex<double>* T_wt_bf_host, LinearSolverType cuda_lin_solver = LinearSolverType::LU);

    ~gw_qpt();

    void init(cublasHandle_t* handle, cusolverDnHandle_t* solver_handle);

    /**
     * \brief synchronous access to a bare polarization array for fixed q, this call makes sure that
     * all asynchronous cuda calculations are finished
     *
     * \param all_done_event cuda synchronization event, shows that all calculation are done and data is available
     * \return Pqk0_tQP array
     */
    cuda_complex* Pqk0_tQP(cudaEvent_t all_done_event);

    /**
     * \brief synchronous access to a dressed polarization array for fixed q, this call makes sure that
     * all asynchronous cuda calculations are finished
     *
     * \param all_done_event cuda synchronization event, shows that all calculation are done and data is available
     * \param calc_stream cuda polarization calculation stream to check for synchronization
     * \param need_minus_q we need to get Pqk_tQP_conj_ for a (-q) point
     * \return Pqk_tQP or Pqk_tQP_conj array for dressed polarization
     */
    cuda_complex* Pqk_tQP(cudaEvent_t all_done_event, cudaStream_t calc_stream, int need_minus_q);

    /**
     * \return synchronization lock to Pqk0_tQP array
     */
    int* Pqk0_tQP_lock();

    /**
     * \brief set Pqk0_tQP to zero
     */
    void reset_Pqk0();

    /**
     * \brief multiply Pkq0_tQP by a scalar
     * \param scale_factor multiplication scalar
     */
    void scale_Pq0_tQP(scalar_t scale_factor);

    /**
     * \brief Dump diagonal elements of P0(q,t) for all tau points to a text file.
     *
     * Output format per line: q_ibz tau diag_index real imag
     *
     * \param file_path path to output text file
     * \param q_ibz irreducible q-point index used as metadata in the dump
     */
    void dump_Pq0_diagonals_to_text(const std::string& file_path, size_t q_ibz);

    /**
     * \brief Solve Bethe-Salpeter equation for dressed Polarization
     */
    void compute_Pq() {
      switch (cuda_lin_solver_) {
        case LinearSolverType::Cholesky:
          compute_Pq_chol();
          break;
        case LinearSolverType::LU:
          compute_Pq_lu();
          break;
      }
    };

    void compute_Pq_chol();
    void compute_Pq_lu();

    /**
     * \brief wait for ther streams to finish loop over k-points
     */
    void wait_for_kpts();

    /**
     * \brief Computes size of the Polarization function for a fixed momentum q in bytes
     *
     * \param nao number of orbitals
     * \param naux size of an auxiliary basis
     * \param nt number of fermionic tau points
     * \param nw_b number of bosonic Matsubara frequencies
     * \return size of the Polarization function
     */
    static std::size_t size(size_t nao, size_t naux, size_t nt, size_t nw_b) {
      return (2 * nt * naux * naux  // Pqk0_tQP
              ) * sizeof(cuda_complex) +
             (naux * naux + 1) * sizeof(int);
    }

    /**
     * \brief Fourier transform of Polarization function from imaginary time to Matsubara frequency
     */
    void transform_tw();

    /**
     * \brief Fourier transform of Polarization function from Matsubara frequency to imaginary time
     */
    void transform_wt();

    /**
     * Stdout verbosity
     * @return verbose level
     */
    int& verbose() { return _verbose; }
    int  verbose() const { return _verbose; }

  private:
    // streams
    cudaStream_t stream_{};
    // streams for potrs
    std::vector<cudaStream_t> streams_potrs_;

    // bare polarization for fixed q and fixed k
    cuda_complex* Pqk0_tQP_;
    cuda_complex* Pqk0_wQP_;
    cuda_complex* Pqk_tQP_;  // these will point to P0. We need them in a  different part of the code, so reuse memory
    cuda_complex* Pqk_wQP_;
    cuda_complex* Pqk_tQP_conj_;
    cuda_complex* T_wt_;
    cuda_complex* T_tw_;
    // events for communicating
    cudaEvent_t              polarization_ready_event_{};
    cudaEvent_t              bare_polarization_ready_event_{};
    cudaEvent_t              Cholesky_decomposition_ready_event_{};
    cudaEvent_t              LU_decomposition_ready_event_{};
    cudaEvent_t              getrs_ready_event_{};
    std::vector<cudaEvent_t> potrs_ready_event_;
    std::vector<cudaEvent_t> one_minus_P_ready_event_;

    // number of atomic orbitals
    const int nao_;
    const int nao2_;
    const int nao3_;
    // number of auxiliary orbitals
    const int naux_;
    const int naux2_;
    const int nauxnao_;
    const int nauxnao2_;
    const int ns_;
    // number of time slices
    const int nt_;
    const int nw_b_;
    const int ntnaux_;
    const int ntnaux2_;
    const int ntnao_;
    const int ntnao2_;
    const int nwnaux_;
    const int nwnaux2_;

    int       _verbose{0};

    // pointer to cublas handle
    cublasHandle_t* handle_;

    // solver internals
    cusolverDnHandle_t* solver_handle_;
    cuda_complex*       one_minus_P_wPQ_;
    cuda_complex**      one_minus_P_w_ptrs_;  // Double pointer for batched potrf
    cuda_complex**      P0_w_ptrs_;           // Double pointer for batched LU
    int*                d_info_{};
    int*                Pivot_{};  // Pivot indices for LU
    // locks so that we don't overwrite P0
    int* Pqk0_tQP_lock_{};

    // CUDA linear solver (pivoted LU or Cholesky)
    const LinearSolverType cuda_lin_solver_;
  };

}  // namespace green::gpu
