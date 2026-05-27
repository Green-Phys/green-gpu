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

#ifndef GREEN_GPU_CU_SYMMETRY_H
#define GREEN_GPU_CU_SYMMETRY_H

#include <cstddef>
#include <complex>
#include <vector>

#include "common_defs.h"
#include "cublas_routines_prec.h"
#include "cuda_types_map.h"

namespace green::gpu {

  /**
   * \brief Plain-data struct carrying all symmetry information needed to initialize cu_symmetry.
   *
   * Built on the CPU side (in g++-compiled code that has access to green-symmetry / HDF5),
   * then passed into cu_symmetry::initialize() so that nvcc never sees HDF5 headers.
   */
  struct cu_symmetry_data {
    size_t nk = 0, ink = 0, nq = 0, inq = 0;

    // k-mesh maps
    std::vector<size_t> k_full_to_reduced;
    std::vector<size_t> k_reduced_to_full;
    std::vector<long>   k_tr_conj;
    std::vector<std::vector<long>> k_stars;   // k_stars[ik_ibz] = list of full-BZ k in star

    // q-mesh maps
    std::vector<size_t> q_full_to_reduced;
    std::vector<size_t> q_reduced_to_full;
    std::vector<long>   q_tr_conj;
    std::vector<std::vector<long>> q_stars;   // q_stars[iq_ibz] = list of full-BZ q in star

    // Momentum-conservation maps, flattened: index = k_full * nq + q_full
    std::vector<size_t> k1_from_k2q_map;   // k1 = k2 + q
    std::vector<size_t> k2_from_k1q_map;   // k2 = k1 - q

    // Pre-built symmetry transforms (leave empty to skip GPU upload).
    // Scalar: nao×nao per k. X2C: nso×nso (= 2nao × 2nao) per k, with σ_y spinor
    // mixing already encoded. Both are uploaded to the device and consumed by the
    // same transform_k_ao_device GEMM pipeline.
    std::vector<std::complex<double>> k_ao_transforms;  // [nk][dim][dim], dim ∈ {nao, nso}
    std::vector<std::complex<double>> q_p0_transforms;  // [nq][naux][naux]
  };


  class cu_symmetry {
  public:
    cu_symmetry() = default;
    cu_symmetry(const cu_symmetry&) = delete;
    cu_symmetry& operator=(const cu_symmetry&) = delete;
    cu_symmetry(cu_symmetry&&) = delete;
    cu_symmetry& operator=(cu_symmetry&&) = delete;

    ~cu_symmetry();

    void initialize(const cu_symmetry_data& data, int nao, int nso, int naux, int nts, int ns);

    bool initialized() const { return initialized_; }

    size_t k_full_to_reduced(size_t k_full) const { return k_full_to_reduced_h_.at(k_full); }
    size_t k_reduced_to_full(size_t k_reduced) const { return k_reduced_to_full_h_.at(k_reduced); }
    size_t q_reduced_to_full(size_t q_reduced) const { return q_reduced_to_full_h_.at(q_reduced); }
    long   q_tr_conj(size_t q_full) const { return q_tr_conj_h_.at(q_full); }
    const std::vector<long>& k_star(size_t ik) const { return k_stars_h_.at(ik); }
    const std::vector<long>& q_star(size_t iq) const { return q_stars_h_.at(iq); }
    size_t k1_from_k2q(size_t k2, size_t q) const { return k1_from_k2q_h_.at(k2 * nq_h_ + q); }
    size_t k2_from_k1q(size_t k1, size_t q) const { return k2_from_k1q_h_.at(k1 * nq_h_ + q); }

    // Return device pointer to (naux × naux) q-space P0 transform matrix for full-BZ q-point q_full.
    // Returns nullptr if q-space transforms were not built.
    const cuDoubleComplex* q_p0_transform_d(size_t q_full) const {
      return q_p0_transform_full_d_ ? q_p0_transform_full_d_ + q_full * naux_ * naux_ : nullptr;
    }
    const cuComplex* q_p0_transform_f(size_t q_full) const {
      return q_p0_transform_full_f_ ? q_p0_transform_full_f_ + q_full * naux_ * naux_ : nullptr;
    }

    // Transform device data: applies U_k * G * U_k^dagger on contiguous matrices of size
    // dim×dim, where dim is the AO-space dimension of the stored k_sym_transform_ao
    // (nao for scalar runs, nso = 2·nao for X2C — the σ_y spinor mixing is baked into
    // the nso×nso U). For TR-related k, conj(U * G * U†) is computed via a single
    // RSCAL on the full output (no per-block manipulation).
    //
    // Layout expectations: in_device and out_device point to nts*ns matrices of size
    // dim*dim each, contiguous and row-major (matching the green-gpu MatrixXcd
    // convention). For X2C callers, ns should be 1 and the four spin blocks of G
    // must be assembled into a single nso×nso block per (k, t) before the call.
    //
    // ibz_in_device: optional alternative source on device (used during chained ops).
    // input_scratch / work_scratch: optional per-caller scratch buffers, each
    // nts*ns*dim*dim elements. When provided, these are used instead of the shared
    // cu_symmetry scratch — enabling concurrent calls from different worker streams.
    void transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuDoubleComplex* in_device, size_t k_full,
                               cuDoubleComplex* out_device, int nts, int ns,
                               cuDoubleComplex* ibz_in_device = nullptr,
                               cuDoubleComplex* input_scratch = nullptr,
                               cuDoubleComplex* work_scratch = nullptr);
    void transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuComplex* in_device, size_t k_full,
                               cuComplex* out_device, int nts, int ns,
                               cuComplex* ibz_in_device = nullptr,
                               cuComplex* input_scratch = nullptr,
                               cuComplex* work_scratch = nullptr);

  private:
    void release();

    bool initialized_ = false;
    int  nao_         = 0;
    int  nso_         = 0;
    int  naux_        = 0;
    int  nts_         = 0;
    int  ns_          = 0;
    size_t batch_count_ = 0;
    size_t matrix_stride_ = 0;

    template <typename cuda_complex_t>
    void transform_k_ao_device_impl(cublasHandle_t handle, cudaStream_t stream, cuda_complex_t* in_device, size_t k_full,
                                    cuda_complex_t* out_device, int nts, int ns, cuda_complex_t* ibz_in_device,
                                    cuda_complex_t* input_scratch, cuda_complex_t* work_scratch);

    size_t*          k_full_to_reduced_d_   = nullptr;
    size_t*          k_reduced_to_full_d_   = nullptr;
    size_t*          q_full_to_reduced_d_   = nullptr;
    size_t*          q_reduced_to_full_d_   = nullptr;
    long*            k_tr_conj_d_           = nullptr;
    long*            q_tr_conj_d_           = nullptr;
    cuDoubleComplex* k_ao_transform_full_d_ = nullptr;
    cuComplex*       k_ao_transform_full_f_ = nullptr;
    cuDoubleComplex* q_p0_transform_full_d_ = nullptr;
    cuComplex*       q_p0_transform_full_f_ = nullptr;

    cuDoubleComplex*      input_batch_z_d_ = nullptr;
    cuDoubleComplex*      work_batch_z_d_  = nullptr;
    cuComplex*            input_batch_f_d_ = nullptr;
    cuComplex*            work_batch_f_d_  = nullptr;

    // Host-side caches populated during initialize(); used by accessor methods
    // and by transform_k_ao_device_impl (k_tr_conj_h_).
    std::vector<size_t>              k_full_to_reduced_h_;
    std::vector<size_t>              k_reduced_to_full_h_;
    std::vector<long>                k_tr_conj_h_;
    std::vector<std::vector<long>>   k_stars_h_;
    std::vector<size_t>              q_reduced_to_full_h_;
    std::vector<long>                q_tr_conj_h_;
    std::vector<std::vector<long>>   q_stars_h_;
    std::vector<size_t>              k1_from_k2q_h_;
    std::vector<size_t>              k2_from_k1q_h_;
    size_t                           nq_h_ = 0;

    int                               k_transform_dim_ = 0;  // sqrt(k_ao_transforms.size()/nk); nao for scalar, nso for X2C
  };
}  // namespace green::gpu

#endif  // GREEN_GPU_CU_SYMMETRY_H
