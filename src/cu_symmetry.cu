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

#include <green/gpu/cu_symmetry.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace green::gpu {
  namespace {
    template <typename to_t, typename from_t>
    void cast_copy_complex(to_t* dst, const from_t* src, size_t count) {
      for (size_t idx = 0; idx < count; ++idx) {
        dst[idx] = static_cast<to_t>(src[idx]);
      }
    }
  }  // namespace

  cu_symmetry::~cu_symmetry() { release(); }

  void cu_symmetry::initialize(const cu_symmetry_data& data, int nao, int naux, int nts, int ns) {
    release();

    const int nk  = static_cast<int>(data.nk);
    const int ink = static_cast<int>(data.ink);
    const int nq  = static_cast<int>(data.nq);
    const int inq = static_cast<int>(data.inq);

    nao_  = nao;
    naux_ = naux;
    nts_  = nts;
    ns_   = ns;
    batch_count_   = static_cast<size_t>(nts) * static_cast<size_t>(ns);
    matrix_stride_ = static_cast<size_t>(nao) * static_cast<size_t>(nao);

    // Cache host-side data for accessor methods and TR-conjugation check
    k_full_to_reduced_h_ = data.k_full_to_reduced;
    k_reduced_to_full_h_ = data.k_reduced_to_full;
    k_tr_conj_h_         = data.k_tr_conj;
    k_stars_h_           = data.k_stars;
    q_reduced_to_full_h_ = data.q_reduced_to_full;
    q_tr_conj_h_         = data.q_tr_conj;
    q_stars_h_           = data.q_stars;
    k1_from_k2q_h_       = data.k1_from_k2q_map;
    k2_from_k1q_h_       = data.k2_from_k1q_map;
    nq_h_                = data.nq;

    auto upload_array = [](auto& dst, const auto& vec, const char* name) {
      using T = typename std::remove_reference_t<decltype(vec)>::value_type;
      if (cudaMalloc(reinterpret_cast<void**>(&dst), vec.size() * sizeof(T)) != cudaSuccess)
        throw std::runtime_error(std::string("Failed to allocate ") + name + ".");
      if (cudaMemcpy(dst, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(std::string("Failed to copy ") + name + " to device.");
    };

    upload_array(k_full_to_reduced_d_, data.k_full_to_reduced, "k_full_to_reduced_d_");
    upload_array(k_reduced_to_full_d_, data.k_reduced_to_full, "k_reduced_to_full_d_");
    upload_array(q_full_to_reduced_d_, data.q_full_to_reduced, "q_full_to_reduced_d_");
    upload_array(q_reduced_to_full_d_, data.q_reduced_to_full, "q_reduced_to_full_d_");
    upload_array(k_tr_conj_d_,         data.k_tr_conj,         "k_tr_conj_d_");
    upload_array(q_tr_conj_d_,         data.q_tr_conj,         "q_tr_conj_d_");

    if (!data.k_ao_transforms.empty()) {
      // Compute transform matrix dimension: nao for scalar, nso (= 2*nao) for X2C.
      size_t dim_sq = data.k_ao_transforms.size() / data.nk;
      k_transform_dim_ = static_cast<int>(std::round(std::sqrt(static_cast<double>(dim_sq))));
      // Upload to GPU only for the scalar (nao×nao) case; X2C (nso×nso) uses transform_k_ao_device_2c instead.
      if (k_transform_dim_ == nao_) {
        std::vector<std::complex<float>> k_ao_f(data.k_ao_transforms.size());
        cast_copy_complex(k_ao_f.data(), data.k_ao_transforms.data(), data.k_ao_transforms.size());

        if (cudaMalloc(&k_ao_transform_full_d_, data.k_ao_transforms.size() * sizeof(cuDoubleComplex)) != cudaSuccess)
          throw std::runtime_error("Failed to allocate k_ao_transform_full_d_.");
        if (cudaMalloc(&k_ao_transform_full_f_, data.k_ao_transforms.size() * sizeof(cuComplex)) != cudaSuccess)
          throw std::runtime_error("Failed to allocate k_ao_transform_full_f_.");
        if (cudaMemcpy(k_ao_transform_full_d_, data.k_ao_transforms.data(), data.k_ao_transforms.size() * sizeof(std::complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
          throw std::runtime_error("Failed to copy k_ao_transform_full to device.");
        if (cudaMemcpy(k_ao_transform_full_f_, k_ao_f.data(), k_ao_f.size() * sizeof(std::complex<float>), cudaMemcpyHostToDevice) != cudaSuccess)
          throw std::runtime_error("Failed to copy float k_ao_transform_full to device.");
      }
    }

    // q_p0_transforms stores U_q row-major (as-is); q_p0_transforms_conj stores
    // conj(U_q) row-major.  Both buffers are consumed by compute_second_tau_contraction,
    // which selects the cuBLAS OP per operand role and TR branch — see the convention
    // block in compute_second_tau_contraction for the OP ↔ math identification.
    // Correctness for complex U_q requires the precomputed conjugate buffer; a
    // single OP on row-major U_q bytes cannot land all four of {U_q, U_q*, U_qᵀ, U_q†}.
    if (!data.q_p0_transforms.empty()) {
      std::vector<std::complex<float>> q_p0_f(data.q_p0_transforms.size());
      cast_copy_complex(q_p0_f.data(), data.q_p0_transforms.data(), data.q_p0_transforms.size());

      const size_t bytes_d = data.q_p0_transforms.size() * sizeof(cuDoubleComplex);
      const size_t bytes_f = data.q_p0_transforms.size() * sizeof(cuComplex);

      if (cudaMalloc(&q_p0_transform_full_d_, bytes_d) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_full_d_.");
      if (cudaMalloc(&q_p0_transform_full_f_, bytes_f) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_full_f_.");
      if (cudaMemcpy(q_p0_transform_full_d_, data.q_p0_transforms.data(), bytes_d, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy q_p0_transform_full to device.");
      if (cudaMemcpy(q_p0_transform_full_f_, q_p0_f.data(), bytes_f, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy float q_p0_transform_full to device.");

      // Allocate parallel conjugate buffers and populate by D2D copy + RSCAL of imag lane.
      if (cudaMalloc(&q_p0_transform_conj_full_d_, bytes_d) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_conj_full_d_.");
      if (cudaMalloc(&q_p0_transform_conj_full_f_, bytes_f) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_conj_full_f_.");
      if (cudaMemcpy(q_p0_transform_conj_full_d_, q_p0_transform_full_d_, bytes_d, cudaMemcpyDeviceToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to D2D-copy q_p0_transform_conj_full_d_.");
      if (cudaMemcpy(q_p0_transform_conj_full_f_, q_p0_transform_full_f_, bytes_f, cudaMemcpyDeviceToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to D2D-copy q_p0_transform_conj_full_f_.");

      // Conjugate via RSCAL on imag lane (stride 2, alpha = -1). Use a fresh handle on the
      // default stream because cu_symmetry::initialize runs before per-worker handles exist.
      cublasHandle_t local_handle;
      if (cublasCreate(&local_handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create local cublas handle for q_p0_transform_conj init.");
      const double minus_one_d = -1.0;
      const float  minus_one_f = -1.0f;
      const int    n_elems = static_cast<int>(data.q_p0_transforms.size());
      const int    two = 2;
      if (RSCAL(local_handle, n_elems, &minus_one_d, reinterpret_cast<double*>(q_p0_transform_conj_full_d_) + 1, two) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(local_handle);
        throw std::runtime_error("RSCAL failed on q_p0_transform_conj_full_d_ init.");
      }
      if (RSCAL(local_handle, n_elems, &minus_one_f, reinterpret_cast<float*>(q_p0_transform_conj_full_f_) + 1, two) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(local_handle);
        throw std::runtime_error("RSCAL failed on q_p0_transform_conj_full_f_ init.");
      }
      cudaDeviceSynchronize();
      cublasDestroy(local_handle);
    }

    if (cudaMalloc(&input_batch_z_d_, batch_count_ * matrix_stride_ * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate input_batch_z_d_.");
    if (cudaMalloc(&work_batch_z_d_, batch_count_ * matrix_stride_ * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate work_batch_z_d_.");
    if (cudaMalloc(&input_batch_f_d_, batch_count_ * matrix_stride_ * sizeof(cuComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate input_batch_f_d_.");
    if (cudaMalloc(&work_batch_f_d_, batch_count_ * matrix_stride_ * sizeof(cuComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate work_batch_f_d_.");

    initialized_ = true;
  }

  void cu_symmetry::release() {
    if (input_batch_z_d_ != nullptr) cudaFree(input_batch_z_d_);
    if (work_batch_z_d_ != nullptr) cudaFree(work_batch_z_d_);
    if (input_batch_f_d_ != nullptr) cudaFree(input_batch_f_d_);
    if (work_batch_f_d_ != nullptr) cudaFree(work_batch_f_d_);
    if (k_ao_transform_full_d_ != nullptr) cudaFree(k_ao_transform_full_d_);
    if (k_ao_transform_full_f_ != nullptr) cudaFree(k_ao_transform_full_f_);
    if (q_p0_transform_full_d_ != nullptr) cudaFree(q_p0_transform_full_d_);
    if (q_p0_transform_full_f_ != nullptr) cudaFree(q_p0_transform_full_f_);
    if (q_p0_transform_conj_full_d_ != nullptr) cudaFree(q_p0_transform_conj_full_d_);
    if (q_p0_transform_conj_full_f_ != nullptr) cudaFree(q_p0_transform_conj_full_f_);
    if (k_full_to_reduced_d_ != nullptr) cudaFree(k_full_to_reduced_d_);
    if (k_reduced_to_full_d_ != nullptr) cudaFree(k_reduced_to_full_d_);
    if (q_full_to_reduced_d_ != nullptr) cudaFree(q_full_to_reduced_d_);
    if (q_reduced_to_full_d_ != nullptr) cudaFree(q_reduced_to_full_d_);
    if (k_tr_conj_d_ != nullptr) cudaFree(k_tr_conj_d_);
    if (q_tr_conj_d_ != nullptr) cudaFree(q_tr_conj_d_);

    input_batch_z_d_ = nullptr;
    work_batch_z_d_ = nullptr;
    input_batch_f_d_ = nullptr;
    work_batch_f_d_ = nullptr;
    k_ao_transform_full_d_ = nullptr;
    k_ao_transform_full_f_ = nullptr;
    q_p0_transform_full_d_ = nullptr;
    q_p0_transform_full_f_ = nullptr;
    q_p0_transform_conj_full_d_ = nullptr;
    q_p0_transform_conj_full_f_ = nullptr;
    k_full_to_reduced_d_ = nullptr;
    k_reduced_to_full_d_ = nullptr;
    q_full_to_reduced_d_ = nullptr;
    q_reduced_to_full_d_ = nullptr;
    k_tr_conj_d_ = nullptr;
    q_tr_conj_d_ = nullptr;
    initialized_ = false;

    k_full_to_reduced_h_.clear();
    k_reduced_to_full_h_.clear();
    k_tr_conj_h_.clear();
    k_stars_h_.clear();
    q_reduced_to_full_h_.clear();
    q_tr_conj_h_.clear();
    q_stars_h_.clear();
    k1_from_k2q_h_.clear();
    k2_from_k1q_h_.clear();
    nq_h_ = 0;
    k_transform_dim_ = 0;
  }

  template <typename cuda_complex_t>
  void cu_symmetry::transform_k_ao_device_impl(cublasHandle_t handle, cudaStream_t stream, cuda_complex_t* in_device, size_t k_full,
                                               cuda_complex_t* out_device, int nts, int ns,
                                               cuda_complex_t* ibz_in_device,
                                               cuda_complex_t* input_scratch, cuda_complex_t* work_scratch) {
    if (!initialized_) throw std::runtime_error("cu_symmetry must be initialized before use.");

    // Use IBZ buffer as input if provided, otherwise use regular input
    cuda_complex_t* source_device = (ibz_in_device != nullptr) ? ibz_in_device : in_device;

    // Copy input from device buffer to work buffer for GEMMs
    size_t batch_elements = static_cast<size_t>(nts) * static_cast<size_t>(ns) * matrix_stride_;

    // Symmetry transform: G(k_full) = U * G(k_ibz) * U†  (non-TR)
    //                      G(k_full) = conj(U * G(k_ibz) * U†)  (TR)
    //
    // Both U and G are stored in ROW-MAJOR (ndarray / green-gpu MatrixXcd convention).
    // CUBLAS interprets row-major data as the transpose of the intended matrix.
    // To compute result_rm = U * G * U†, we need result_cm = (U*G*U†)^T = U^* * G^T * U^T.
    // With row-major U: OP_C → (U_rm^T)^H = U_rm^*, OP_N → U_rm^T.
    // With row-major G: OP_N → G_rm^T.
    // So: GEMM1(OP_C, OP_N) = U^* * G^T, GEMM2(OP_N, OP_N) = result * U^T.

    if constexpr (std::is_same_v<cuda_complex_t, cuDoubleComplex>) {
      cuDoubleComplex* input_buf = input_scratch ? input_scratch : input_batch_z_d_;
      cuDoubleComplex* work_buf  = work_scratch  ? work_scratch  : work_batch_z_d_;

      cudaMemcpyAsync(input_buf, source_device, batch_elements * sizeof(cuda_complex_t), cudaMemcpyDeviceToDevice, stream);

      cublasSetStream(handle, stream);
      const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
      const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
      const cuDoubleComplex* U = k_ao_transform_full_d_ + k_full * matrix_stride_;

      // work = U^* * G^T  (col-major intermediate)
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_C, CUBLAS_OP_N, nao_, nao_, nao_, &one, U, nao_, 0, input_buf, nao_,
                               matrix_stride_, &zero, work_buf, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed first batched GEMM in cu_symmetry::transform_k_ao_device_impl.");
      // out = work * U^T  (col-major result, row-major interpretation = U * G * U†)
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nao_, &one, work_buf, nao_, matrix_stride_, U,
                               nao_, 0, &zero, out_device, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed second batched GEMM in cu_symmetry::transform_k_ao_device_impl.");

      // TR: conjugate output to get conj(U * G_ibz * U†)
      if (k_tr_conj_h_.at(k_full) != 0) {
        const double minus_one = -1.0;
        if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<double*>(out_device) + 1, 2) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("Failed TR conjugation in cu_symmetry::transform_k_ao_device_impl.");
      }
    } else {
      cuComplex* input_buf = input_scratch ? input_scratch : input_batch_f_d_;
      cuComplex* work_buf  = work_scratch  ? work_scratch  : work_batch_f_d_;

      cudaMemcpyAsync(input_buf, source_device, batch_elements * sizeof(cuda_complex_t), cudaMemcpyDeviceToDevice, stream);

      cublasSetStream(handle, stream);
      const cuComplex one = make_cuComplex(1.0f, 0.0f);
      const cuComplex zero = make_cuComplex(0.0f, 0.0f);
      const cuComplex* U = k_ao_transform_full_f_ + k_full * matrix_stride_;

      // work = U^* * G^T  (col-major intermediate)
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_C, CUBLAS_OP_N, nao_, nao_, nao_, &one, U, nao_, 0, input_buf, nao_,
                               matrix_stride_, &zero, work_buf, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed first batched GEMM in cu_symmetry::transform_k_ao_device_impl.");
      // out = work * U^T  (col-major result, row-major interpretation = U * G * U†)
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nao_, &one, work_buf, nao_, matrix_stride_, U,
                               nao_, 0, &zero, out_device, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed second batched GEMM in cu_symmetry::transform_k_ao_device_impl.");

      // TR: conjugate output to get conj(U * G_ibz * U†)
      if (k_tr_conj_h_.at(k_full) != 0) {
        const float minus_one = -1.0f;
        if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<float*>(out_device) + 1, 2) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("Failed TR conjugation in cu_symmetry::transform_k_ao_device_impl.");
      }
    }
  }

  void cu_symmetry::transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuDoubleComplex* in_device, size_t k_full,
                                          cuDoubleComplex* out_device, int nts, int ns,
                                          cuDoubleComplex* ibz_in_device,
                                          cuDoubleComplex* input_scratch,
                                          cuDoubleComplex* work_scratch) {
    transform_k_ao_device_impl(handle, stream, in_device, k_full, out_device, nts, ns, ibz_in_device, input_scratch, work_scratch);
  }

  void cu_symmetry::transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuComplex* in_device, size_t k_full,
                                          cuComplex* out_device, int nts, int ns,
                                          cuComplex* ibz_in_device,
                                          cuComplex* input_scratch,
                                          cuComplex* work_scratch) {
    transform_k_ao_device_impl(handle, stream, in_device, k_full, out_device, nts, ns, ibz_in_device, input_scratch, work_scratch);
  }

  template <typename cuda_complex_t>
  void cu_symmetry::transform_k_ao_device_2c_impl(cublasHandle_t handle, cudaStream_t stream,
                                                  cuda_complex_t* ibz_in_device, size_t k_full,
                                                  cuda_complex_t* out_device, int nts, int nao) {
    using scalar_t = std::conditional_t<std::is_same_v<cuda_complex_t, cuDoubleComplex>, double, float>;
    const size_t block_elems = static_cast<size_t>(nts) * nao * nao;
    const size_t block_bytes = block_elems * sizeof(cuda_complex_t);

    if (k_tr_conj_h_.at(k_full) == 0) {
      // No TR: copy all 4 blocks unchanged.
      cudaMemcpyAsync(out_device, ibz_in_device, 4 * block_bytes, cudaMemcpyDeviceToDevice, stream);
    } else {
      // TR needed (hardcoded minus_t=true block permutation):
      //   ss=0 <- +conj(ibz ss=1)   [aa <- conj(bb)]
      //   ss=1 <- +conj(ibz ss=0)   [bb <- conj(aa)]
      //   ss=2 <- -conj(ibz ss=2)   [self]
      //   ss=3 <- -conj(ibz ss=3)   [self]
      // ibz_in_device and out_device are distinct, so ss=0/1 swap is safe.
      cudaMemcpyAsync(out_device + 0 * block_elems, ibz_in_device + 1 * block_elems, block_bytes, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(out_device + 1 * block_elems, ibz_in_device + 0 * block_elems, block_bytes, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(out_device + 2 * block_elems, ibz_in_device + 2 * block_elems, block_bytes, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(out_device + 3 * block_elems, ibz_in_device + 3 * block_elems, block_bytes, cudaMemcpyDeviceToDevice, stream);

      cublasSetStream(handle, stream);
      const scalar_t minus_one = static_cast<scalar_t>(-1.0);
      // ss=0, ss=1: +conj -> negate imaginary parts only
      if (RSCAL(handle, static_cast<int>(block_elems), &minus_one,
                reinterpret_cast<scalar_t*>(out_device + 0 * block_elems) + 1, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("RSCAL conj ss=0 failed in transform_k_ao_device_2c.");
      if (RSCAL(handle, static_cast<int>(block_elems), &minus_one,
                reinterpret_cast<scalar_t*>(out_device + 1 * block_elems) + 1, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("RSCAL conj ss=1 failed in transform_k_ao_device_2c.");
      // ss=2, ss=3: -conj -> negate real parts only
      if (RSCAL(handle, static_cast<int>(block_elems), &minus_one,
                reinterpret_cast<scalar_t*>(out_device + 2 * block_elems) + 0, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("RSCAL -conj ss=2 failed in transform_k_ao_device_2c.");
      if (RSCAL(handle, static_cast<int>(block_elems), &minus_one,
                reinterpret_cast<scalar_t*>(out_device + 3 * block_elems) + 0, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("RSCAL -conj ss=3 failed in transform_k_ao_device_2c.");
    }
  }

  void cu_symmetry::transform_k_ao_device_2c(cublasHandle_t handle, cudaStream_t stream,
                                              cuDoubleComplex* ibz_in_device, size_t k_full,
                                              cuDoubleComplex* out_device, int nts, int nao) {
    transform_k_ao_device_2c_impl(handle, stream, ibz_in_device, k_full, out_device, nts, nao);
  }

  void cu_symmetry::transform_k_ao_device_2c(cublasHandle_t handle, cudaStream_t stream,
                                              cuComplex* ibz_in_device, size_t k_full,
                                              cuComplex* out_device, int nts, int nao) {
    transform_k_ao_device_2c_impl(handle, stream, ibz_in_device, k_full, out_device, nts, nao);
  }

}  // namespace green::gpu