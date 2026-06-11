/*
 * Symmetry transform roundtrip test:
 *   Fock_fbz(k_full) -> pick IBZ rep -> transform_k_ao_device -> compare to original
 *
 * Also provides a CPU reference using Eigen to diagnose CUBLAS convention issues.
 */

#include <green/gpu/cu_symmetry.h>

#include <complex>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace green::gpu {

  // CPU reference: U * M * U† with optional TR conjugation of result
  static void cpu_transform(const std::complex<double>* U, const std::complex<double>* M_ibz,
                            std::complex<double>* M_out, int nao, bool tr_conj) {
    // temp = U * M_ibz
    std::vector<std::complex<double>> temp(nao * nao, 0.0);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        for (int a = 0; a < nao; ++a)
          temp[i * nao + j] += U[i * nao + a] * M_ibz[a * nao + j];
    // out = temp * U†
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) {
        std::complex<double> acc = 0.0;
        for (int b = 0; b < nao; ++b)
          acc += temp[i * nao + b] * std::conj(U[j * nao + b]);
        M_out[i * nao + j] = tr_conj ? std::conj(acc) : acc;
      }
  }

  template <class prec>
  double test_symmetry_transform_roundtrip(
      const cu_symmetry_data& sym_data,
      const std::complex<double>* Fock_fbz,
      int ns, int nk, int nao) {

    using cuda_complex = typename cu_type_map<std::complex<prec>>::cuda_type;
    using cxx_complex  = typename cu_type_map<std::complex<prec>>::cxx_type;

    int naosq = nao * nao;

    cu_symmetry<prec> sym;
    sym.initialize(sym_data, nao, /*naux=*/0, /*nts=*/1, /*ns=*/1);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasCreate failed in symmetry test");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    cuda_complex* d_in  = nullptr;
    cuda_complex* d_out = nullptr;
    cudaMalloc(&d_in,  naosq * sizeof(cuda_complex));
    cudaMalloc(&d_out, naosq * sizeof(cuda_complex));

    double max_err_gpu = 0.0;
    double max_err_cpu = 0.0;
    double max_err_gpu_vs_cpu = 0.0;
    int worst_k_gpu = -1, worst_s_gpu = -1;
    int worst_k_cpu = -1, worst_s_cpu = -1;

    const auto* U_rm = sym_data.k_ao_transforms.data();
    std::vector<std::complex<double>> result_host(naosq);
    std::vector<std::complex<double>> cpu_result(naosq);
    std::vector<cxx_complex>          F_ibz_T(naosq);
    std::vector<cxx_complex>          result_T(naosq);

    for (int s = 0; s < ns; ++s) {
      for (int k_full = 0; k_full < nk; ++k_full) {
        size_t k_ibz   = sym_data.k_full_to_reduced[k_full];
        bool   tr_conj = (sym_data.k_tr_conj[k_full] != 0);
        const auto* F_ibz = Fock_fbz + (s * nk + sym_data.k_reduced_to_full[k_ibz]) * naosq;
        const auto* F_ref = Fock_fbz + (s * nk + k_full) * naosq;

        // Stage F_ibz into the right precision (cast if prec=float)
        if constexpr (std::is_same_v<prec, double>) {
          cudaMemcpy(d_in, F_ibz, naosq * sizeof(cuda_complex), cudaMemcpyHostToDevice);
        } else {
          for (int i = 0; i < naosq; ++i)
            F_ibz_T[i] = static_cast<cxx_complex>(F_ibz[i]);
          cudaMemcpy(d_in, F_ibz_T.data(), naosq * sizeof(cuda_complex), cudaMemcpyHostToDevice);
        }

        sym.transform_k_ao_device(handle, stream, d_in, k_full, d_out, /*nts=*/1, /*ns=*/1);
        cudaStreamSynchronize(stream);

        // Copy back and cast to double for comparison
        if constexpr (std::is_same_v<prec, double>) {
          cudaMemcpy(result_host.data(), d_out, naosq * sizeof(cuda_complex), cudaMemcpyDeviceToHost);
        } else {
          cudaMemcpy(result_T.data(), d_out, naosq * sizeof(cuda_complex), cudaMemcpyDeviceToHost);
          for (int i = 0; i < naosq; ++i)
            result_host[i] = static_cast<std::complex<double>>(result_T[i]);
        }

        for (int ij = 0; ij < naosq; ++ij) {
          double err = std::abs(result_host[ij] - F_ref[ij]);
          if (err > max_err_gpu) { max_err_gpu = err; worst_k_gpu = k_full; worst_s_gpu = s; }
        }
        cpu_transform(U_rm + k_full * naosq, F_ibz, cpu_result.data(), nao, tr_conj);
        for (int ij = 0; ij < naosq; ++ij) {
          double err = std::abs(cpu_result[ij] - F_ref[ij]);
          if (err > max_err_cpu) { max_err_cpu = err; worst_k_cpu = k_full; worst_s_cpu = s; }
          double diff = std::abs(result_host[ij] - cpu_result[ij]);
          if (diff > max_err_gpu_vs_cpu) max_err_gpu_vs_cpu = diff;
        }
      }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    std::cout << "Symmetry transform roundtrip ("
              << (std::is_same_v<prec, float> ? "float" : "double") << "):\n"
              << "  GPU vs ref max error: " << max_err_gpu << " at k=" << worst_k_gpu << " s=" << worst_s_gpu << "\n"
              << "  CPU vs ref max error: " << max_err_cpu << " at k=" << worst_k_cpu << " s=" << worst_s_cpu << "\n"
              << "  GPU vs CPU max error: " << max_err_gpu_vs_cpu << "\n";

    return max_err_gpu;
  }

  // Explicit instantiation for both precisions used by the test driver.
  template double test_symmetry_transform_roundtrip<float>(
      const cu_symmetry_data&, const std::complex<double>*, int, int, int);
  template double test_symmetry_transform_roundtrip<double>(
      const cu_symmetry_data&, const std::complex<double>*, int, int, int);

}  // namespace green::gpu
