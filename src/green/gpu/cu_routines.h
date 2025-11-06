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

#ifndef GREEN_GPU_CU_ROUTINES_H
#define GREEN_GPU_CU_ROUTINES_H
#include <green/gpu/common_defs.h>

#include <cstring>

#include "cublas_routines_prec.h"
#include "cuda_common.h"
#include "cugw_qpt.h"
#include "df_integral_types_e.h"

__global__ void initialize_array(cuDoubleComplex* array, cuDoubleComplex value, int count);

namespace green::gpu {
  using hf_reader1 = std::function<void(int, int, std::complex<double>*, ztensor<4>&)>;
  using hf_reader2 = std::function<void(int, int, ztensor<4>&)>;

  template <typename prec>
  using gw_reader1_callback =
      std::function<void(int, int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&, std::complex<double>*,
                         tensor<std::complex<prec>, 4>&, tensor<std::complex<prec>, 4>&, bool, bool)>;
  template <typename prec>
  using gw_reader2_callback = std::function<void(int, int, int, const std::array<size_t, 4>&, tensor<std::complex<prec>, 3>&,
                                                 std::complex<double>*, tensor<std::complex<prec>, 4>&, bool)>;

  using irre_pos_callback   = std::function<size_t(size_t)>;
  using mom_cons_callback   = std::function<const std::array<size_t, 4>(const std::array<size_t, 3>&)>;

  /**
   * \brief utilities to run Hartree-Fock on GPUs
   */
  class cuhf_utils {
    using scalar_t     = typename cu_type_map<std::complex<double>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<double>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<double>>::cuda_type;

  public:
    cuhf_utils(size_t nk, size_t ink, size_t ns, size_t nao, size_t NQ, size_t nkbatch, ztensor<4> dm_fbz, int _myid,
               int _intranode_rank, int _devCount_per_node);

    ~cuhf_utils();

    static std::size_t size_divided_by_kbatch(size_t nao, size_t naux) {
      return (4 * naux * nao * nao + nao * nao) * sizeof(cuda_complex);
    }

    /**
     * \brief Compute exchange part of the Hartree-Fock self-energy
     * \param Vk1k2_Qij
     * \param V_kbatchQij
     * \param new_Fock - static part of the self-energy
     * \param _nk_batch - number of k-batches
     * \param integral_type - type of integral (stored as a whole or need to be read by chunks)
     * \param devices_rank - MPI rank of current process in the device communicator
     * \param devices_size - size of the device communicator
     * \param irre_list - list of reduced points
     * \param r1 - callback function to obtain required part of Coulomb integral from a shared-memory stored integral
     * \param r2 - callback function to obtain required part of Coulomb integral from a localy stored integral
     */
    void solve(std::complex<double>* Vk1k2_Qij, ztensor<4>& V_kbatchQij, ztensor<4>& new_Fock, int _nk_batch,
               integral_reading_type integral_type, int devices_rank, int devices_size, const std::vector<size_t>& irre_list,
               hf_reader1& r1, hf_reader2& r2);

  private:
    /**
     * \brief Prepare and copy integrals on a GPU device to evaluate exchange diagram
     * VkbatchQij_host = V_k(k2~k2+nk_batch, NQ, nao, nao)
     * \param VkbatchQij_host integrals stored on the host
     * \param k_pos outer k-point
     * \param k2 inner k-point
     */
    void set_up_exchange(cxx_complex* VkbatchQij_host, std::size_t k_pos, std::size_t k2);

    /**
     * \brief compute exchage diagram on a GPU device for specific k-points
     */
    void   compute_exchange_diagram();

    bool   _X2C;
    size_t _nao;
    size_t _NQ;
    size_t _naosq;
    size_t _NQnaosq;
    size_t _ns;
    size_t _nk;
    size_t _ink;
    size_t _nkbatch;
    size_t _k2;
    size_t _k_pos;
    // Global objects
    cuda_complex* _Dm_fbz_sk2ba;
    cuda_complex* _F_skij;
    cuda_complex* _weights_fbz;
    // Intermediate objects
    cuda_complex* _VkbatchQij;
    cuda_complex* _VkbatchaQj_conj;
    cuda_complex* _X_kbatchQij;
    cuda_complex* _X_kbatchiaQ;
    cuda_complex* _Y_kbatchij;

    // Pinned host memory for interaction matrix
    cxx_complex* _V_kQij_buffer;

    cudaStream_t _stream;
    // pointer to cublas handle
    cublasHandle_t _handle;
  };

  template <typename prec>
  class cugw_utils {
    using scalar_t     = typename cu_type_map<std::complex<prec>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;

  public:
    cugw_utils(int _nts, int _nt_batch, int _nw_b, int _ns, int _nk, int _ink, int _nqkpt, int _NQ, int _nao,
               ztensor_view<5>& G_tskij_host, bool _low_device_memory, const MatrixXcd& Ttn_FB, const MatrixXcd& Tnt_BF,
               LinearSolverType cuda_lin_solver, int _myid, int _intranode_rank, int _devCount_per_node);

    ~cugw_utils();

    void solve(int _nts, int _ns, int _nk, int _ink, int _nao, const std::vector<size_t>& reduced_to_full,
               const std::vector<size_t>& full_to_reduced, std::complex<double>* Vk1k2_Qij, ztensor<5>& Sigma_tskij_host,
               int _devices_rank, int _devices_size, bool _low_device_memory, int verbose, irre_pos_callback& irre_pos,
               mom_cons_callback& momentum_conservation, gw_reader1_callback<prec>& r1, gw_reader2_callback<prec>& r2);

  private:
    void copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij, int k, int nts, int ns);
    void copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k, int nts);

    //

    bool                           _X2C;
    bool                           _low_device_memory;
    cublasHandle_t                 _handle;
    cusolverDnHandle_t             _solver_handle;
    std::vector<cublasHandle_t>    _qkpt_cublas_handles;  // list of cublas handles for qkpt streams

    gw_qpt<prec>                   qpt;
    std::vector<gw_qkpt<prec>*>    qkpts;

    tensor<std::complex<prec>, 3>  V_Qpm;
    tensor<std::complex<prec>, 3>  V_Qim;
    tensor<std::complex<prec>, 4>  Gk1_stij;
    tensor<std::complex<prec>, 4>  Gk_smtij;
    tensor<std::complex<prec>, 4>& Sigmak_stij = Gk_smtij;

    cuda_complex*                  g_kstij_device;
    cuda_complex*                  g_ksmtij_device;
    cuda_complex*                  sigma_kstij_device;

    int*                           sigma_k_locks;
  };
}  // namespace green::gpu

#endif  // GREEN_GPU_CU_ROUTINES_H
