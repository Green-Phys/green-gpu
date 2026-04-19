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

#include <cstring>
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

    if (cudaMalloc(&k_full_to_reduced_d_, data.k_full_to_reduced.size() * sizeof(size_t)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate k_full_to_reduced_d_.");
    if (cudaMalloc(&k_reduced_to_full_d_, data.k_reduced_to_full.size() * sizeof(size_t)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate k_reduced_to_full_d_.");
    if (cudaMalloc(&q_full_to_reduced_d_, data.q_full_to_reduced.size() * sizeof(size_t)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate q_full_to_reduced_d_.");
    if (cudaMalloc(&q_reduced_to_full_d_, data.q_reduced_to_full.size() * sizeof(size_t)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate q_reduced_to_full_d_.");
    if (cudaMalloc(&k_tr_conj_d_, data.k_tr_conj.size() * sizeof(long)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate k_tr_conj_d_.");
    if (cudaMalloc(&q_tr_conj_d_, data.q_tr_conj.size() * sizeof(long)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate q_tr_conj_d_.");

    if (cudaMemcpy(k_full_to_reduced_d_, data.k_full_to_reduced.data(), data.k_full_to_reduced.size() * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy k_full_to_reduced to device.");
    if (cudaMemcpy(k_reduced_to_full_d_, data.k_reduced_to_full.data(), data.k_reduced_to_full.size() * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy k_reduced_to_full to device.");
    if (cudaMemcpy(q_full_to_reduced_d_, data.q_full_to_reduced.data(), data.q_full_to_reduced.size() * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy q_full_to_reduced to device.");
    if (cudaMemcpy(q_reduced_to_full_d_, data.q_reduced_to_full.data(), data.q_reduced_to_full.size() * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy q_reduced_to_full to device.");
    if (cudaMemcpy(k_tr_conj_d_, data.k_tr_conj.data(), data.k_tr_conj.size() * sizeof(long), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy k_tr_conj_list to device.");
    if (cudaMemcpy(q_tr_conj_d_, data.q_tr_conj.data(), data.q_tr_conj.size() * sizeof(long), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to copy q_tr_conj_list to device.");

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

    if (!data.q_p0_transforms.empty()) {
      std::vector<std::complex<float>> q_p0_f(data.q_p0_transforms.size());
      cast_copy_complex(q_p0_f.data(), data.q_p0_transforms.data(), data.q_p0_transforms.size());

      if (cudaMalloc(&q_p0_transform_full_d_, data.q_p0_transforms.size() * sizeof(cuDoubleComplex)) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_full_d_.");
      if (cudaMalloc(&q_p0_transform_full_f_, data.q_p0_transforms.size() * sizeof(cuComplex)) != cudaSuccess)
        throw std::runtime_error("Failed to allocate q_p0_transform_full_f_.");
      if (cudaMemcpy(q_p0_transform_full_d_, data.q_p0_transforms.data(), data.q_p0_transforms.size() * sizeof(std::complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy q_p0_transform_full to device.");
      if (cudaMemcpy(q_p0_transform_full_f_, q_p0_f.data(), q_p0_f.size() * sizeof(std::complex<float>), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy float q_p0_transform_full to device.");
    }

    if (cudaMallocHost(&host_batch_z_, batch_count_ * matrix_stride_ * sizeof(std::complex<double>)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate host_batch_z_.");
    if (cudaMallocHost(&host_batch_f_, batch_count_ * matrix_stride_ * sizeof(std::complex<float>)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate host_batch_f_.");
    if (cudaMalloc(&input_batch_z_d_, batch_count_ * matrix_stride_ * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate input_batch_z_d_.");
    if (cudaMalloc(&work_batch_z_d_, batch_count_ * matrix_stride_ * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate work_batch_z_d_.");
    if (cudaMalloc(&input_batch_f_d_, batch_count_ * matrix_stride_ * sizeof(cuComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate input_batch_f_d_.");
    if (cudaMalloc(&work_batch_f_d_, batch_count_ * matrix_stride_ * sizeof(cuComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate work_batch_f_d_.");

    device_view_double_.nk  = nk;
    device_view_double_.ink = ink;
    device_view_double_.nq  = nq;
    device_view_double_.inq = inq;
    device_view_double_.nao  = nao;
    device_view_double_.naux = naux;
    device_view_double_.k_full_to_reduced_d = k_full_to_reduced_d_;
    device_view_double_.k_reduced_to_full_d = k_reduced_to_full_d_;
    device_view_double_.k_tr_conj_d         = k_tr_conj_d_;
    device_view_double_.q_full_to_reduced_d = q_full_to_reduced_d_;
    device_view_double_.q_reduced_to_full_d = q_reduced_to_full_d_;
    device_view_double_.q_tr_conj_d         = q_tr_conj_d_;
    device_view_double_.k_ao_transform_full_d = k_ao_transform_full_d_;
    device_view_double_.q_p0_transform_full_d = q_p0_transform_full_d_;

    device_view_float_.nk  = nk;
    device_view_float_.ink = ink;
    device_view_float_.nq  = nq;
    device_view_float_.inq = inq;
    device_view_float_.nao  = nao;
    device_view_float_.naux = naux;
    device_view_float_.k_full_to_reduced_d = k_full_to_reduced_d_;
    device_view_float_.k_reduced_to_full_d = k_reduced_to_full_d_;
    device_view_float_.k_tr_conj_d         = k_tr_conj_d_;
    device_view_float_.q_full_to_reduced_d = q_full_to_reduced_d_;
    device_view_float_.q_reduced_to_full_d = q_reduced_to_full_d_;
    device_view_float_.q_tr_conj_d         = q_tr_conj_d_;
    device_view_float_.k_ao_transform_full_d = k_ao_transform_full_f_;
    device_view_float_.q_p0_transform_full_d = q_p0_transform_full_f_;

    initialized_ = true;
  }

  void cu_symmetry::release() {
    if (input_batch_z_d_ != nullptr) cudaFree(input_batch_z_d_);
    if (work_batch_z_d_ != nullptr) cudaFree(work_batch_z_d_);
    if (input_batch_f_d_ != nullptr) cudaFree(input_batch_f_d_);
    if (work_batch_f_d_ != nullptr) cudaFree(work_batch_f_d_);
    if (host_batch_z_ != nullptr) cudaFreeHost(host_batch_z_);
    if (host_batch_f_ != nullptr) cudaFreeHost(host_batch_f_);
    if (k_ao_transform_full_d_ != nullptr) cudaFree(k_ao_transform_full_d_);
    if (k_ao_transform_full_f_ != nullptr) cudaFree(k_ao_transform_full_f_);
    if (q_p0_transform_full_d_ != nullptr) cudaFree(q_p0_transform_full_d_);
    if (q_p0_transform_full_f_ != nullptr) cudaFree(q_p0_transform_full_f_);
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
    host_batch_z_ = nullptr;
    host_batch_f_ = nullptr;
    k_ao_transform_full_d_ = nullptr;
    k_ao_transform_full_f_ = nullptr;
    q_p0_transform_full_d_ = nullptr;
    q_p0_transform_full_f_ = nullptr;
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
                                               cuda_complex_t* ibz_in_device) {
    if (!initialized_) throw std::runtime_error("cu_symmetry must be initialized before use.");

    // Use IBZ buffer as input if provided, otherwise use regular input
    cuda_complex_t* source_device = (ibz_in_device != nullptr) ? ibz_in_device : in_device;

    // Copy input from device buffer to work buffer for GEMMs
    size_t batch_elements = static_cast<size_t>(nts) * static_cast<size_t>(ns) * matrix_stride_;

    // The Green's function is stored in row-major (ndarray), but CUBLAS works in
    // column-major. CUBLAS sees G_cm = G^T. To produce the correct physical transform
    // G_full = U * G * U† in row-major, the column-major result must be:
    //   non-TR: result_cm = U^* * G_cm * U^T
    //   TR:     result_cm = U * conj(G_cm) * U^H
    // We achieve this by: (1) conjugate input, (2) apply U * X * U^H GEMMs,
    // (3) conjugate output only for non-TR.
    // Proof: conj(U * conj(G_cm) * U^H) = U^* * G_cm * U^T (non-TR case).

    if constexpr (std::is_same_v<cuda_complex_t, cuDoubleComplex>) {
      cudaMemcpyAsync(input_batch_z_d_, source_device, batch_elements * sizeof(cuda_complex_t), cudaMemcpyDeviceToDevice, stream);

      cublasSetStream(handle, stream);
      const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
      const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
      const double minus_one = -1.0;
      const cuDoubleComplex* U = k_ao_transform_full_d_ + k_full * matrix_stride_;

      // Conjugate input: G_cm -> conj(G_cm)
      if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<double*>(input_batch_z_d_) + 1, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed input conjugation in cu_symmetry::transform_k_ao_device_impl.");

      // Compute U * conj(G_cm) * U^H
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nao_, &one, U, nao_, 0, input_batch_z_d_, nao_,
                               matrix_stride_, &zero, work_batch_z_d_, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed first batched GEMM in cu_symmetry::transform_k_ao_device_impl.");
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_C, nao_, nao_, nao_, &one, work_batch_z_d_, nao_, matrix_stride_, U,
                               nao_, 0, &zero, out_device, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed second batched GEMM in cu_symmetry::transform_k_ao_device_impl.");

      // For non-TR: conjugate output to get U^* * G_cm * U^T
      // For TR: U * conj(G_cm) * U^H is already the correct result
      if (k_tr_conj_h_.at(k_full) == 0) {
        if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<double*>(out_device) + 1, 2) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("Failed output conjugation in cu_symmetry::transform_k_ao_device_impl.");
      }
    } else {
      cudaMemcpyAsync(input_batch_f_d_, source_device, batch_elements * sizeof(cuda_complex_t), cudaMemcpyDeviceToDevice, stream);

      cublasSetStream(handle, stream);
      const cuComplex one = make_cuComplex(1.0f, 0.0f);
      const cuComplex zero = make_cuComplex(0.0f, 0.0f);
      const float minus_one = -1.0f;
      const cuComplex* U = k_ao_transform_full_f_ + k_full * matrix_stride_;

      // Conjugate input: G_cm -> conj(G_cm)
      if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<float*>(input_batch_f_d_) + 1, 2) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed input conjugation in cu_symmetry::transform_k_ao_device_impl.");

      // Compute U * conj(G_cm) * U^H
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nao_, &one, U, nao_, 0, input_batch_f_d_, nao_,
                               matrix_stride_, &zero, work_batch_f_d_, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed first batched GEMM in cu_symmetry::transform_k_ao_device_impl.");
      if (GEMM_STRIDED_BATCHED(handle, CUBLAS_OP_N, CUBLAS_OP_C, nao_, nao_, nao_, &one, work_batch_f_d_, nao_, matrix_stride_, U,
                               nao_, 0, &zero, out_device, nao_, matrix_stride_, static_cast<int>(nts * ns)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed second batched GEMM in cu_symmetry::transform_k_ao_device_impl.");

      // For non-TR: conjugate output to get U^* * G_cm * U^T
      // For TR: U * conj(G_cm) * U^H is already the correct result
      if (k_tr_conj_h_.at(k_full) == 0) {
        if (RSCAL(handle, static_cast<int>(batch_elements), &minus_one, reinterpret_cast<float*>(out_device) + 1, 2) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("Failed output conjugation in cu_symmetry::transform_k_ao_device_impl.");
      }
    }
  }

  void cu_symmetry::transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuDoubleComplex* in_device, size_t k_full,
                                          cuDoubleComplex* out_device, int nts, int ns,
                                          cuDoubleComplex* ibz_in_device) {
    transform_k_ao_device_impl(handle, stream, in_device, k_full, out_device, nts, ns, ibz_in_device);
  }

  void cu_symmetry::transform_k_ao_device(cublasHandle_t handle, cudaStream_t stream, cuComplex* in_device, size_t k_full,
                                          cuComplex* out_device, int nts, int ns,
                                          cuComplex* ibz_in_device) {
    transform_k_ao_device_impl(handle, stream, in_device, k_full, out_device, nts, ns, ibz_in_device);
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