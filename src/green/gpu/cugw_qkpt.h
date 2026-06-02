/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "common_defs.h"
#include "cublas_routines_prec.h"
#include "cuda_common.h"

namespace green::gpu {

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
     * \brief Setup device memory to compute bare polarization bubble
     *
     * For X2C low-memory: pass host G pointers; data is staged through pinned buffers.
     * For high-memory scalar: pass nullptr for both G pointers; D2D copies from pre-loaded device buffers.
     */
    void upload_p0_inputs(cxx_complex* Gk1_stij_host, cxx_complex* Gk_smtij_host, cxx_complex* V_Qpm_host, int k, int k1);

    /**
     * \brief Upload Coulomb integrals only (low-memory scalar path).
     *
     * G(k) and G(k1) are loaded separately by the caller via load_Gk1_to_device()
     * and cu_symmetry::transform_k_ao_device().
     */
    void upload_p0_coulomb(cxx_complex* V_Qpm_host, int k, int k1);

    /**
     * \brief Setup device memory to compute G-W contraction and obtain self-energy
     *
     * \param Gk1_stij_host Host stored Green's function for a specific k-point
     * \param V_Qim_host Host stored three-center Coulomb integral
     * \param k first k-point (IBZ index)
     * \param k1 second k-point (IBZ index)
     */
    void upload_sigma_inputs(cxx_complex* Gk1_stij_host, cxx_complex* V_Qim_host, int k, int k1);

    /**
     * \brief Load G(k1) from host to device via pinned staging buffer (low-memory scalar mode)
     *
     * Copies host data into per-worker pinned buffer, then stages to device asynchronously.
     * Safe against concurrent workers overwriting the unpinned source.
     *
     * \param host_ptr Host source pointer (unpinned)
     * \param n_elems Number of complex elements to copy
     */
    void load_Gk1_to_device(cxx_complex* host_ptr, size_t n_elems);

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
     * \param Pqk_tQP Dressed polarization bubble
     */
    void compute_second_tau_contraction(cuda_complex* Pqk_tQP = nullptr, const cuda_complex* U_q = nullptr,
                                         bool q_conj_after_uq = false);

    /**
     * \brief Using dressed GW polarization compute self-energy at a given momentum point (X2C version)
     *
     * \param Pqk_tQP Dressed polarization bubble
     */
    // X2C self-energy contraction.  Implements the chain (in CPU math notation):
    //
    //       Y2_out = U_dag · P · U · Y1ᵀ            (then Σ = -V_nPj · Y2_out)
    //
    // via three GEMMs.  All three row-major-stored matrices (U, U_dag, Pq) are
    // "r2c-converted" through cuBLAS OP_T; the final outermost factor U_dag uses
    // OP_N on a buffer whose col-major view already IS the adjoint of the math U.
    //
    // The caller supplies BOTH role-named operands, plus the Pq matching the
    // chosen branch:
    //
    //   non-TR:  Pq    = Pqk_tQP_       (P_qdeg = U_q · P · U_q†)
    //            U     = U_q stored row-major
    //            U_dag = U_q_conj stored row-major   ← OP_N(this) = X† = U_q† math
    //
    //   TR:      Pq    = Pqk_tQP_conj_  (P_qdeg = U_q_conj · conj(P) · U_q_conj†)
    //            U     = U_q_conj stored row-major
    //            U_dag = U_q stored row-major        ← OP_N(this) = Xᵀ = U_q_conj† math
    //
    // The kernel is branch-free: fixed OPs (OP_T at 2a on U, OP_T at 2b on Pq,
    // OP_N at 2c on U_dag) apply to both branches.  See src/cugw_qkpt.cu for the
    // row/col-major convention derivation.
    void compute_second_tau_contraction_2C(cuda_complex* Pqk_tQP = nullptr,
                                           const cuda_complex* U = nullptr,
                                           const cuda_complex* U_dag = nullptr);

    /**
     * \brief For a given k-point copy self-energy back to a host memory
     * \param low_memory_mode - whether the whole self-energy allocated in memory or not
     */
    void write_sigma(bool low_memory_mode = false);

    /**
     * \brief return the status of copy_selfenergy from device to host.
     * Return value of `false` means stream is ready for next calculation without cleanup.
     *
     * \return true if cleanup is required, false otherwise
     */
    bool require_cleanup(){
      return cleanup_req_;
    }

    /**
     * \brief perform cleanup, i.e. copy data from Sigmak buffer (4-index array for a given momentum point) to Host shared memory Self-energy
     *
     * \param low_memory_mode - whether the whole self-energy allocated in memory or not
     * \param Sigmak_stij_host - Host stored self-energy object at a given momentum point
     * \param Sigma_tskij_host - Host stored full self-energy object for each device
     * \param x2c - use x2c specific functions or not
     */
    void cleanup(bool low_memory_mode, tensor<std::complex<prec>, 4>& Sigmak_stij_host, ztensor<5>& Sigma_tskij_host, bool x2c);

    /**
     * \brief Copy the non-relativistic self-energy from the per-k buffer to the full host tensor.
     *
     * \param Sigma_tskij_host Host storage for the full self-energy tensor.
     * \param Sigmak_stij Self-energy tensor for a given momentum point.
     */
    void copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij);

    /**
     * \brief Copy 2-component self-energy from per-k buffer to full host tensor.
     *
     * \param Sigma_tskij_host Host storage for the full self-energy tensor.
     * \param Sigmak_stij Self-energy tensor for a given momentum point.
     */
    void copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij);

    /**
     * \brief Check if cuda devices are busy
     * \return true if asynchronous calculations are still running
     */
    bool is_busy();

    /**
     * \brief Set the irreducible k-point index for gw_qkpt worker
     *
     * \param k incoming k-point index
     */
    void set_k_red_id(int k) {
      k_red_id_ = k;
    }

    static std::size_t size(size_t nao, size_t naux, size_t nt, size_t nt_batch, size_t ns) {
      return (2 * naux * nao * nao               // V_Qpm+V_pmQ
              + naux * naux * nt_batch           // local copy of P
              + 2 * nt_batch * naux * nao * nao  // X1 and X2
              + 3 * ns * nt * nao * nao          // sigmak_stij, g_stij, g_smtij
              + 2 * ns * nt * nao * nao          // transform_input_scratch, transform_work_scratch
              ) *
             sizeof(cuda_complex);
    }

    cudaEvent_t  all_done_event() const { return all_done_event_; }
    cudaEvent_t  data_ready_event() const { return data_ready_event_; }
    cudaEvent_t  transform_done_event() const { return transform_done_event_; }
    cudaStream_t stream() const { return stream_; }
    cublasHandle_t handle() const { return *handle_; }
    cuda_complex* g_stij_device() const { return g_stij_; }
    cuda_complex* g_smtij_device() const { return g_smtij_; }

    // Per-worker scratch buffers for symmetry transforms.
    // These replace the shared cu_symmetry scratch to enable concurrent transforms
    // from different worker streams without data races.
    cuda_complex* transform_input_scratch() const { return transform_input_scratch_; }
    cuda_complex* transform_work_scratch() const { return transform_work_scratch_; }

  private:
    // Upload V_Qpm to device and build V_pmQ = conj(V_Qpm^T); shared by upload_p0_inputs/upload_p0_coulomb.
    void upload_coulomb_v_first(cxx_complex* V_Qpm_host);
    // Upload V_Qim to device and build V_nPj = V_Qim†; shared by upload_sigma_inputs.
    void upload_coulomb_v_second(cxx_complex* V_Qim_host);

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
    // Pinned host memory for Gk_stij and Gk_smtij
    cxx_complex* Gk1_stij_buffer_;
    cxx_complex* Gk_smtij_buffer_;
    cxx_complex* Sigmak_stij_buffer_;

    // Per-worker scratch for cu_symmetry::transform_k_ao_device (each ns*nt*nao*nao elements)
    cuda_complex* transform_input_scratch_{nullptr};
    cuda_complex* transform_work_scratch_{nullptr};

    // events for communicating
    cudaEvent_t data_ready_event_;    // this event is recorded once data has arrived and computation can be started
    cudaEvent_t all_done_event_;      // this event is recorded after we're all done with the computation
    cudaEvent_t transform_done_event_{}; // fired after the first transform_k_ao_device; ibz_g_device_ is free after this

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

    // irreducible k-pt assigned to the qkpt stream
    int k_red_id_;

    // status of data transfer / copy from Device to Host.
    // false: not required, stream ready for next calculation
    // true: required, stream occupied
    bool cleanup_req_ = false;
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

  /**
   * \brief returns an idle qkpt stream, otherwise waits until a stream is available
   *
   * \tparam prec - precision for calculation
   * \param qkpts - vector of qkpt workers (gw_qkpt<prec> type)
   * \param low_memory_mode - low memory mode for read/write integrals
   * \param Sigmak_stij_host - cudaMallocHost buffer for transfering Sigma
   * \return gw_qkpt<prec>* - pointer to idle qkpt
   */
  template <typename prec>
  gw_qkpt<prec>* obtain_idle_qkpt_for_sigma(std::vector<gw_qkpt<prec>*>& qkpts, bool low_memory_mode,
                                            tensor<std::complex<prec>,4>& Sigmak_stij_host,
                                            ztensor<5>& Sigma_tskij_host, bool x2c) {
    static int pos = 0;
    pos++;
    if (pos >= qkpts.size()) pos = 0;
    while (qkpts[pos]->is_busy()) {
      pos = (pos + 1) % qkpts.size();
    }
    qkpts[pos]->cleanup(low_memory_mode, Sigmak_stij_host, Sigma_tskij_host, x2c);
    return qkpts[pos];
  }

  /**
   * \brief waits for all qkpts to complete and cleans them up
   *
   * \tparam prec - precision for calculation
   * \param qkpts - vector of qkpt workers (gw_qkpt<prec> type)
   * \param low_memory_mode - low memory mode for read/write integrals
   * \param Sigmak_stij_host - cudaMallocHost buffer for transfering Sigma
   */
  template <typename prec>
  void wait_and_clean_qkpts(std::vector<gw_qkpt<prec>*>& qkpts, bool low_memory_mode,
                            tensor<std::complex<prec>,4>& Sigmak_stij_host,
                            ztensor<5>& Sigma_tskij_shared, bool x2c) {
    for (int pos = 0; pos < qkpts.size(); pos++) {
      // wait for qkpt to finish its tasks, then cleanup
      while (qkpts[pos]->is_busy()) {
        continue;
      }
      qkpts[pos]->cleanup(low_memory_mode, Sigmak_stij_host, Sigma_tskij_shared, x2c);
    }
  }

}  // namespace green::gpu
