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

#ifndef GREEN_GPU_CUHF_UTILS_H
#define GREEN_GPU_CUHF_UTILS_H
#include <green/gpu/common_defs.h>

#include <cstring>
#include <functional>
#include <vector>

#include "cublas_routines_prec.h"
#include "cuda_common.h"
#include "df_integral_types_e.h"

__global__ void initialize_array(cuDoubleComplex* array, cuDoubleComplex value, int count);

namespace green::gpu {
  using hf_reader1 = std::function<void(int, int, std::complex<double>*, ztensor<4>&)>;
  using hf_reader2 = std::function<void(int, int, ztensor<4>&)>;

  /**
   * \brief utilities to run Hartree-Fock on GPUs
   */
  class cuhf_utils {
    using scalar_t     = typename cu_type_map<std::complex<double>>::cxx_base_type;
    using cxx_complex  = typename cu_type_map<std::complex<double>>::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<double>>::cuda_type;

  public:
    // Scalar (ns=1,2) and legacy X2C (ns=3, 3-block host dm_fbz layout) constructor.
    cuhf_utils(size_t nk, size_t ink, size_t ns, size_t nao, size_t NQ, size_t nkbatch, ztensor<4> dm_fbz, int _myid,
               int _intranode_rank, int _devCount_per_node);

    // X2C constructor with device-resident dm_fbz in (nk, nso, nso) row-major layout.
    // The per-spin-block GEMMs in add_exchange_to_fock pick aa/bb/ab sub-views with
    // lda=nso and a per-ss (row, col) offset; ba is later derived as ab.adjoint() in
    // copy_2c_Fock_from_device_to_host. Caller retains ownership of dm_fbz_nso_device;
    // the data is copied into an internal device buffer at construction.
    cuhf_utils(size_t nk, size_t ink, size_t nao, size_t NQ, size_t nkbatch,
               const cuDoubleComplex* dm_fbz_nso_device,
               int _myid, int _intranode_rank, int _devCount_per_node);

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
    void accumulate_exchange_on_device(std::complex<double>* Vk1k2_Qij, ztensor<4>& V_kbatchQij, ztensor<4>& new_Fock,
                       int _nk_batch, integral_reading_type integral_type, int devices_rank, int devices_size,
                       const std::vector<size_t>& irre_list, hf_reader1& r1, hf_reader2& r2);

  private:
    void set_up_exchange(cxx_complex* VkbatchQij_host, std::size_t k_pos, std::size_t k2);
    void add_exchange_to_fock();

    bool   _X2C;
    size_t _nao;
    size_t _nso;     // 2*nao for X2C, == _nao for scalar (unused)
    size_t _nsosq;   // _nso * _nso (X2C device-layout stride)
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
    cuda_complex* _Dm_fbz_sk2ba = nullptr;  // legacy 3-block (s, k2, b, a) layout
    cuda_complex* _Dm_fbz_nso   = nullptr;  // X2C device-layout (k, nso, nso); non-null selects X2C path
    cuda_complex* _F_skij;
    cuda_complex* _weights_fbz;
    // Intermediate objects
    cuda_complex* _VkbatchQij;
    cuda_complex* _VkbatchaQj_conj;
    cuda_complex* _X_kbatchQij;
    cuda_complex* _X_kbatchiaQ;
    cuda_complex* _Y_kbatchij;
    // Pinned host memory for interaction matrix
    cxx_complex*   _V_kQij_buffer;
    cudaStream_t   _stream;
    cublasHandle_t _handle;
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_CUHF_UTILS_H
