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

#ifndef GREEN_GPU_CUGW_UTILS_H
#define GREEN_GPU_CUGW_UTILS_H
#include <green/gpu/common_defs.h>

#include <cstring>
#include <functional>
#include <vector>

#include "cublas_routines_prec.h"
#include "cu_symmetry.h"
#include "cuda_common.h"
#include "cugw_qkpt.h"
#include "cugw_qpt.h"

namespace green::gpu {

  template <typename prec>
  using gw_reader0_callback = std::function<void(int, tensor<std::complex<prec>, 4>&)>;

  // gw_reader1_callback: Read Coulomb integrals and G(k1) for first tau contraction
  // Parameters:
  //   k, k1: full-BZ k-points
  //   k_reduced_id, k1_reduced_id: IBZ indices
  //   k_vector: {k, 0, q, k+q} for integral reading
  //   V_Qpm: output Coulomb integrals (Qpm format)
  //   Vk1k2_Qij: auxiliary buffer for integral symmetrization
  //   Gk1_stij: output G(k1) Green's function
  //   need_minus_k, need_minus_k1: time-reversal flags (used in X2C and high-memory modes; always false in low-memory scalar)
  template <typename prec>
  using gw_reader1_callback =
      std::function<void(int, int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&, std::complex<double>*,
                         tensor<std::complex<prec>, 4>&, bool, bool)>;

  // gw_reader2_callback: Read Coulomb integrals and G(k1) for second tau contraction
  // Parameters:
  //   k, k1: full-BZ k-points
  //   k1_reduced_id: IBZ index
  //   k_vector: {k, q, 0, k-q} for integral reading
  //   V_Qim: output Coulomb integrals (Qim format)
  //   Vk1k2_Qij: auxiliary buffer for integral symmetrization
  //   Gk1_stij: output G(k1) Green's function
  //   need_minus_k1: time-reversal flag (used in X2C and high-memory modes; always false in low-memory scalar)
  template <typename prec>
  using gw_reader2_callback = std::function<void(int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&,
                                                 std::complex<double>*, tensor<std::complex<prec>, 4>&, bool)>;

  template <typename prec>
  class cugw_utils {
    using scalar_t     = typename cu_type_map<std::complex<prec>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;

  public:
    cugw_utils(int _nts, int _nt_batch, int _nw_b, int _ns, int _nk, int _ink, int _nq, int _inq, int _nqkpt, int _NQ, int _nao, int _nso,
               const cu_symmetry_data& sym_data, ztensor_view<5>& G_tskij_host, bool _low_device_memory,
               const MatrixXcd& Ttn_FB, const MatrixXcd& Tnt_BF,
               LinearSolverType cuda_lin_solver, int _myid, int _intranode_rank, int _devCount_per_node);

    ~cugw_utils();

    void accumulate_gw_selfenergy_on_device(int _nts, int _ns, int _nk, int _ink, int _nq, int _inq, int _nao,
              std::complex<double>* Vk1k2_Qij, ztensor<5>& Sigma_tskij_host,
                                            int _devices_rank, int _devices_size, bool _low_device_memory, int verbose,
                                            gw_reader0_callback<prec>& r0,
                                            gw_reader1_callback<prec>& r1,
                                            gw_reader2_callback<prec>& r2);

  private:
    void copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij, int k, int nts, int ns);
    void copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k, int nts);

    // P0 build helpers — one per execution mode
    void prepare_first_contraction_highmem_scalar(gw_qkpt<prec>* qkpt, size_t k_full, size_t k1_full);
    void prepare_first_contraction_lowmem_scalar(gw_qkpt<prec>* qkpt, size_t k_full, size_t k1_full);

    // Sigma accumulation helpers — one per execution mode
    void accumulate_sigma_scalar(gw_qkpt<prec>* qkpt, size_t k1, size_t q_deg, bool q_need_conj);
    void accumulate_sigma_x2c(gw_qkpt<prec>* qkpt, size_t q_deg, bool q_need_conj);

    // IBZ G upload with stream fencing (low-memory scalar path)
    void upload_ibz_g(size_t k_ibz_id);

    bool                           _X2C;
    bool                           _low_device_memory;
    cublasHandle_t                 _handle;
    cusolverDnHandle_t             _solver_handle;
    std::vector<cublasHandle_t>    _qkpt_cublas_handles;

    gw_qpt<prec>                   qpt;
    std::vector<gw_qkpt<prec>*>    qkpts;
    ztensor_view<5>&               G_tskij_host_;

    tensor<std::complex<prec>, 3>  V_Qpm;
    tensor<std::complex<prec>, 3>  V_Qim;
    tensor<std::complex<prec>, 4>  Gk1_stij;
    tensor<std::complex<prec>, 4>  Gk_smtij;
    tensor<std::complex<prec>, 4>& Sigmak_stij = Gk_smtij;

    cuda_complex*                  g_kstij_device;
    cuda_complex*                  g_ksmtij_device;
    cuda_complex*                  sigma_kstij_device;

    cu_symmetry                    _cu_symmetry;

    int*                           sigma_k_locks;

    // IBZ upload infrastructure: dedicated stream + GPU-side fencing
    cudaStream_t                   ibz_upload_stream_{};
    cudaEvent_t                    ibz_upload_ready_event_{};
    cxx_complex*                   ibz_pinned_buffer_{nullptr};
    cuda_complex*                  ibz_g_device_{nullptr};       // shared device buffer for G(k_ibz,-tau)
    std::vector<cudaEvent_t>       prev_epoch_events_;
    size_t                         ibz_g_elems_{0};              // ns * nts * nao * nao
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_CUGW_UTILS_H
