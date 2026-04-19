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
#include <type_traits>
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

    // Pre-built symmetry transforms (leave empty to skip GPU upload)
    // For scalar: k_ao_transforms stores nao×nao matrices per k-point.
    // For X2C: k_ao_transforms stores nso×nso (= 2nao × 2nao) matrices per k-point;
    //          these are used for CPU-side transforms only (no GPU upload).
    std::vector<std::complex<double>> k_ao_transforms;  // [nk][nao_or_nso][nao_or_nso]
    std::vector<std::complex<double>> q_p0_transforms;  // [nq][naux][naux]
  };


  /**
   * \brief Lightweight symmetry view intended to reside on CUDA device memory.
   *
   * The class stores raw device pointers to compact symmetry metadata and
   * provides device-callable transforms that mirror the core behavior needed
   * from green-symmetry:
   * 1) k and q mesh mapping info (full <-> reduced, nk/ink/nq/inq)
   * 2) AO-space ibz -> full-k transform (k_sym_transform_ao-like)
   * 3) P0-space ibz -> full-q transform (q_sym_transform_p0-like)
   *
   * Transform matrices are expected in row-major flattened storage:
   * - k_ao_transform_full: [nk][nao][nao]
   * - q_p0_transform_full: [nq][naux][naux]
   */
  template <typename cxx_complex_t>
  struct cu_device_symmetry_handler {
    using cu_map_t     = cu_type_map<cxx_complex_t>;
    using cu_complex_t = typename cu_map_t::cuda_type;
    using scalar_t     = typename cu_map_t::cxx_base_type;

    // Mesh sizes
    size_t nk  = 0;
    size_t ink = 0;
    size_t nq  = 0;
    size_t inq = 0;

    // Optional dimensions for transform methods
    int nao  = 0;
    int naux = 0;

    // k-mesh maps on device
    const size_t* k_full_to_reduced_d = nullptr;
    const size_t* k_reduced_to_full_d = nullptr;
    const long*   k_tr_conj_d         = nullptr;

    // q-mesh maps on device
    const size_t* q_full_to_reduced_d = nullptr;
    const size_t* q_reduced_to_full_d = nullptr;
    const long*   q_tr_conj_d         = nullptr;

    // Optional symmetry transforms stored per full k/q index
    const cu_complex_t* k_ao_transform_full_d = nullptr;  // [nk][nao][nao]
    const cu_complex_t* q_p0_transform_full_d = nullptr;  // [nq][naux][naux]

    __host__ __device__ size_t k_reduced_from_full(size_t k_full) const {
      return (k_full_to_reduced_d != nullptr) ? k_full_to_reduced_d[k_full] : k_full;
    }

    __host__ __device__ size_t k_full_from_reduced(size_t k_reduced) const {
      return (k_reduced_to_full_d != nullptr) ? k_reduced_to_full_d[k_reduced] : k_reduced;
    }

    __host__ __device__ size_t q_reduced_from_full(size_t q_full) const {
      return (q_full_to_reduced_d != nullptr) ? q_full_to_reduced_d[q_full] : q_full;
    }

    __host__ __device__ size_t q_full_from_reduced(size_t q_reduced) const {
      return (q_reduced_to_full_d != nullptr) ? q_reduced_to_full_d[q_reduced] : q_reduced;
    }

    __host__ __device__ static cu_complex_t conj_complex(cu_complex_t z) {
      return cu_map_t::cast(cu_map_t::real(z), -cu_map_t::imag(z));
    }

    __host__ __device__ static cu_complex_t add_mul(cu_complex_t acc, cu_complex_t a, cu_complex_t b) {
      return cu_map_t::add(acc, cu_map_t::mul(a, b));
    }

    /**
     * \brief Transform AO matrix from IBZ representative to full k-point.
     *
     * out = U_k * X * U_k^dagger, with optional conjugation of X for
     * time-reversal-related points indicated by k_tr_conj_d.
     */
    __device__ void k_sym_transform_ao(cu_complex_t* out, const cu_complex_t* in_ibz, int k_full) const {
      const int dim       = nao;
      const bool need_conj = (k_tr_conj_d != nullptr) && (k_tr_conj_d[k_full] != 0);

      if (k_ao_transform_full_d == nullptr) {
        for (int i = 0; i < dim * dim; ++i) {
          out[i] = need_conj ? conj_complex(in_ibz[i]) : in_ibz[i];
        }
        return;
      }

      const cu_complex_t* U = k_ao_transform_full_d + static_cast<size_t>(k_full) * dim * dim;
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          cu_complex_t acc = cu_map_t::cast(0.0, 0.0);
          for (int a = 0; a < dim; ++a) {
            const cu_complex_t Uia = U[static_cast<size_t>(i) * dim + a];
            for (int b = 0; b < dim; ++b) {
              cu_complex_t Xab = in_ibz[static_cast<size_t>(a) * dim + b];
              if (need_conj) Xab = conj_complex(Xab);
              const cu_complex_t Ujb_dag = conj_complex(U[static_cast<size_t>(j) * dim + b]);
              acc                       = add_mul(acc, cu_map_t::mul(Uia, Xab), Ujb_dag);
            }
          }
          out[static_cast<size_t>(i) * dim + j] = acc;
        }
      }
    }

    /**
     * \brief Transform P0-like auxiliary matrix from IBZ representative to full q-point.
     *
     * out = S_q * P0 * S_q^dagger, with optional conjugation of P0 for
     * time-reversal-related points indicated by q_tr_conj_d.
     *
     * NOTE: Reserved for q-space symmetry transforms in second tau contraction (q-star loop).
     * Currently not used; q_star polarization transformations are handled via RSCAL conjugation.
     */
    __device__ void q_sym_transform_p0(cu_complex_t* out, const cu_complex_t* in_ibz, int q_full) const {
      const int dim       = naux;
      const bool need_conj = (q_tr_conj_d != nullptr) && (q_tr_conj_d[q_full] != 0);

      if (q_p0_transform_full_d == nullptr) {
        for (int i = 0; i < dim * dim; ++i) {
          out[i] = need_conj ? conj_complex(in_ibz[i]) : in_ibz[i];
        }
        return;
      }

      const cu_complex_t* S = q_p0_transform_full_d + static_cast<size_t>(q_full) * dim * dim;
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          cu_complex_t acc = cu_map_t::cast(0.0, 0.0);
          for (int a = 0; a < dim; ++a) {
            const cu_complex_t Sia = S[static_cast<size_t>(i) * dim + a];
            for (int b = 0; b < dim; ++b) {
              cu_complex_t Pab = in_ibz[static_cast<size_t>(a) * dim + b];
              if (need_conj) Pab = conj_complex(Pab);
              const cu_complex_t Sjb_dag = conj_complex(S[static_cast<size_t>(j) * dim + b]);
              acc                       = add_mul(acc, cu_map_t::mul(Sia, Pab), Sjb_dag);
            }
          }
          out[static_cast<size_t>(i) * dim + j] = acc;
        }
      }
    }
  };

  class cu_symmetry {
  public:
    cu_symmetry() = default;
    cu_symmetry(const cu_symmetry&) = delete;
    cu_symmetry& operator=(const cu_symmetry&) = delete;
    cu_symmetry(cu_symmetry&&) = delete;
    cu_symmetry& operator=(cu_symmetry&&) = delete;

    ~cu_symmetry();

    void initialize(const cu_symmetry_data& data, int nao, int naux, int nts, int ns);

    bool initialized() const { return initialized_; }

    template <typename CxxComplexT>
    const cu_device_symmetry_handler<CxxComplexT>& device_view() const {
      if constexpr (std::is_same_v<CxxComplexT, std::complex<double>>)
        return device_view_double_;
      else
        return device_view_float_;
    }

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

    // Transform device data already allocated (e.g., in qkpt buffers)
    // Applies U_k * G * U_k^dagger transform on device with optional time-reversal conjugation from symmetry
    // Can optionally use a separate IBZ buffer as input (for transforming IBZ representative)
    // Time-reversal conjugation (if needed) must be done on host side in r0/r1 callback before loading to device
    void transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuDoubleComplex* in_device, size_t k_full,
                               cuDoubleComplex* out_device, int nts, int ns,
                               cuDoubleComplex* ibz_in_device = nullptr);
    void transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuComplex* in_device, size_t k_full,
                               cuComplex* out_device, int nts, int ns,
                               cuComplex* ibz_in_device = nullptr);

    // Device-side X2C TR spin-flip for G(k_ibz, -tau) → G(k_full, -tau).
    // Input ibz_in_device holds the IBZ Green's function in 4-block layout [4, nts, nao, nao].
    // If k_full is the IBZ representative (no TR needed), performs a direct device-to-device copy.
    // If k_full is an anti-unitary image of its IBZ rep, applies the hardcoded minus_t=true
    // block permutation + sign/conjugation: ss=0↔ss=1 with conj, ss=2,3 self with -conj.
    // ibz_in_device and out_device must be distinct buffers.
    void transform_k_ao_device_2c(cublasHandle_t handle, cudaStream_t stream,
                                   cuDoubleComplex* ibz_in_device, size_t k_full,
                                   cuDoubleComplex* out_device, int nts, int nao);
    void transform_k_ao_device_2c(cublasHandle_t handle, cudaStream_t stream,
                                   cuComplex* ibz_in_device, size_t k_full,
                                   cuComplex* out_device, int nts, int nao);

  private:
    void release();

    bool initialized_ = false;
    int  nao_         = 0;
    int  naux_        = 0;
    int  nts_         = 0;
    int  ns_          = 0;
    size_t batch_count_ = 0;
    size_t matrix_stride_ = 0;

    cu_device_symmetry_handler<std::complex<double>> device_view_double_{};
    cu_device_symmetry_handler<std::complex<float>>  device_view_float_{};

    template <typename cuda_complex_t>
    void transform_k_ao_device_impl(cublasHandle_t handle, cudaStream_t stream, cuda_complex_t* in_device, size_t k_full,
                                    cuda_complex_t* out_device, int nts, int ns, cuda_complex_t* ibz_in_device);

    template <typename cuda_complex_t>
    void transform_k_ao_device_2c_impl(cublasHandle_t handle, cudaStream_t stream,
                                       cuda_complex_t* ibz_in_device, size_t k_full,
                                       cuda_complex_t* out_device, int nts, int nao);

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

    std::complex<double>* host_batch_z_ = nullptr;
    std::complex<float>*  host_batch_f_ = nullptr;
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
