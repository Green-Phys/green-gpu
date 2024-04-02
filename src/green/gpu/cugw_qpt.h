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
#include <iostream>
#include <vector>

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
    gw_qpt(int nao, int naux, int nt, int nw_b, const std::complex<double>* T_tw_fb_host,
           const std::complex<double>* T_wt_bf_host, int verbose);

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
     * \brief Solve Bethe-Salpeter equation for dressed Polarization
     */
    void compute_Pq();

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
    cudaStream_t stream_;
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
    cudaEvent_t              polarization_ready_event_;
    cudaEvent_t              bare_polarization_ready_event_;
    cudaEvent_t              Cholesky_decomposition_ready_event_;
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
    int*                d_info_;

    // locks so that we don't overwrite P0
    int* Pqk0_tQP_lock_;
  };

  template <typename prec>
  class gw_qkpt {
    using scalar_t     = typename cu_type_map<std::complex<prec>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;

  public:
    gw_qkpt(int nao, int naux, int ns, int nt, int nt_batch, cublasHandle_t* handle, cuda_complex* g_ktij, cuda_complex* g_kmtij,
            cuda_complex* sigma_ktij, int* sigma_k_locks);

    ~gw_qkpt();

    /**
     * \brief Setup device memory to compute bare  polarization bubble
     *
     * \param Gk1_stij_host Host stored Green's function for specific k-point for positive imaginary time
     * \param Gk_smtij_host Host stored Green's function for specific k-point for negative imaginary time
     * \param V_Qpm_host Host stored three-center Coulomb integral for specific k-pair
     * \param k - first k-point
     * \param need_minus_k - we actually need -k (apply symmetry relations)
     * \param k1 - second k-point
     * \param need_minus_k1 - we actually need -k1 (apply symmetry relations)
     */
    void set_up_qkpt_first(cxx_complex* Gk1_stij_host, cxx_complex* Gk_smtij_host, cxx_complex* V_Qpm_host, int k,
                           bool need_minus_k, int k1, bool need_minus_k1);

    /**
     * \brief Setup device memory to compute G-W contraction and obtain self-energy
     *
     * \param Gk1_stij_host Host stored Green's function for a specific k-point
     * \param V_Qim_host Host stored three-center Coulomb integral
     * \param k first k-point
     * \param k1 second k-point
     * \param need_minus_k1 we actually need -k (apply symmetry relations)
     */
    void set_up_qkpt_second(cxx_complex* Gk1_stij_host, cxx_complex* V_Qim_host, int k, int k1, bool need_minus_k1);

    /**
     * \brief Perform first contraction VG(tau)VG(-tau)
     *
     * \param Pqk0_tQP - output buffer for bare polarization bubble
     * \param Pqk0_tQP_lock synchronization lock for bare polarization bubble
     */
    void compute_first_tau_contraction(cuda_complex* Pqk0_tQP, int* Pqk0_tQP_lock);

    /**
     * \brief Write locally computed polarization for a given imaginary time point into a full array
     *
     * \param t current imaginary time point
     * \param Pqk0_tQP bare polarization bubble
     * \param Pqk0_tQP_lock synchronization lock for polarization
     */
    void write_P0(int t, cuda_complex* Pqk0_tQP, int* Pqk0_tQP_lock);

    /**
     * \brief Using dressed GW polarization compute self-energy at a given momentum point
     *
     * \param Sigmak_stij_host Host stored array for Self-energy at a given momentum point
     * \param Pqk_tQP Dressed polarization bubble
     */
    void compute_second_tau_contraction(cxx_complex* Sigmak_stij_host = nullptr, cuda_complex* Pqk_tQP = nullptr);
    /**
     * \brief Using dressed GW polarization compute self-energy at a given momentum point (X2C version)
     * \param Sigmak_stij_host Host stored array for Self-energy at a given momentum point
     * \param Pqk_tQP Dressed polarization bubble
     */
    void compute_second_tau_contraction_2C(cxx_complex* Sigmak_stij_host = nullptr, cuda_complex* Pqk_tQP = nullptr);

    /**
     * \brief For a given k-point copy self-energy back to a host memory
     * \param low_memory_mode - whether the whole self-energy allocated in memory or not
     * \param Sigmak_stij_host - Host stored self-energy object at a given momentum point
     */
    void write_sigma(bool low_memory_mode = false, cxx_complex* Sigmak_stij_host = nullptr);

    /**
     * \brief Check if cuda devices are budy
     * \return true if asynchronous calculations are still running
     */
    bool is_busy();

    //
    static std::size_t size(size_t nao, size_t naux, size_t nt, size_t nt_batch, size_t ns) {
      return (2 * naux * nao * nao               // V_Qpm+V_pmQ
              + naux * naux * nt_batch           // local copy of P
              + 2 * nt_batch * naux * nao * nao  // X1 and X2
              + 3 * ns * nt * nao * nao          // sigmak_stij, g_stij, g_smtij
              ) *
             sizeof(cuda_complex);
    }

    cudaEvent_t  all_done_event() const { return all_done_event_; }
    cudaStream_t stream() const { return stream_; }

  private:
    bool _low_memory_requirement;
    // externally handled/allocated Green's functions and self-energies
    cuda_complex* g_ktij_;
    cuda_complex* g_kmtij_;
    cuda_complex* sigma_ktij_;
    int*          sigma_k_locks_;
    /// streams
    cudaStream_t stream_;

    // Interaction matrix, density decomposed.
    cuda_complex* V_Qpm_;
    cuda_complex* V_pmQ_;

    // these are two aliases to V matrices to avoid alloc the second time around.
    cuda_complex* V_Qim_;
    cuda_complex* V_nPj_;

    // intermediate vars for strided batched multiplies
    cuda_complex* g_stij_;
    cuda_complex* g_smtij_;
    cuda_complex* X2t_Ptm_;
    cuda_complex* X1t_tmQ_;
    cuda_complex* Pqk0_tQP_local_;

    // intermediate vars for temp storage of sigma
    cuda_complex* sigmak_stij_;
    // Pinned host memory for interaction matrix
    cxx_complex* V_Qpm_buffer_;
    // Pineed host memory for Gk_stij and Gk_smtij
    cxx_complex* Gk1_stij_buffer_;
    cxx_complex* Gk_smtij_buffer_;
    cxx_complex* Sigmak_stij_buffer_;

    // events for communicating
    cudaEvent_t data_ready_event_;  // this event is recorded once data has arrived and computation can be started
    cudaEvent_t all_done_event_;    // this event is recorded after we're all done with the computation

    // number of k-points
    // const int nk_;
    // number of atomic orbitals
    const int nao_;
    const int nao2_;
    const int nao3_;
    // number of auxiliary orbitals
    const int naux_;
    const int naux2_;
    const int nauxnao_;
    const int nauxnao2_;
    // number of spins
    const int ns_;
    // number of time slices
    const int nt_;
    const int ntnaux_;
    const int ntnaux2_;
    const int ntnao_;
    const int ntnao2_;

    const int nt_batch_;

    // momentum indices
    int k_;
    int k1_;

    // lock to make sure we're not overwriting P0
    int* Pqk0_tQP_lock_;

    // pointer to cublas handle
    cublasHandle_t* handle_;
  };

  template <typename prec>
  gw_qkpt<prec>* obtain_idle_qkpt(std::vector<gw_qkpt<prec>*>& qkpts) {
    static int pos = 0;
    pos++;
    if (pos >= qkpts.size()) pos = 0;
    while (qkpts[pos]->is_busy()) {
      pos = (pos + 1) % qkpts.size();
    }
    return qkpts[pos];
  }

}  // namespace green::gpu