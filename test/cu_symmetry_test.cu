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

  double test_symmetry_transform_roundtrip(
      const cu_symmetry_data& sym_data,
      const std::complex<double>* Fock_fbz,  // [ns, nk, nao, nao] row-major
      int ns, int nk, int nao) {

    int ink = static_cast<int>(sym_data.k_reduced_to_full.size());
    int naosq = nao * nao;

    // Initialize cu_symmetry on device
    cu_symmetry sym;
    sym.initialize(sym_data, nao, /*naux=*/0, /*nts=*/1, /*ns=*/1);

    // CUDA/CUBLAS setup
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasCreate failed in symmetry test");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    // Allocate device buffers for one nao×nao matrix (nts=1, ns=1)
    cuDoubleComplex* d_in  = nullptr;
    cuDoubleComplex* d_out = nullptr;
    cudaMalloc(&d_in,  naosq * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, naosq * sizeof(cuDoubleComplex));

    double max_err_gpu = 0.0;
    double max_err_cpu = 0.0;
    double max_err_gpu_vs_cpu = 0.0;
    int worst_k_gpu = -1, worst_s_gpu = -1;
    int worst_k_cpu = -1, worst_s_cpu = -1;

    // U matrices in k_ao_transforms are already row-major
    // (MatrixXcd = Eigen::RowMajor in green-gpu), use directly.
    const auto* U_rm = sym_data.k_ao_transforms.data();

    std::vector<std::complex<double>> result_host(naosq);
    std::vector<std::complex<double>> cpu_result(naosq);

    for (int s = 0; s < ns; ++s) {
      for (int k_full = 0; k_full < nk; ++k_full) {
        size_t k_ibz = sym_data.k_full_to_reduced[k_full];
        bool tr_conj = (sym_data.k_tr_conj[k_full] != 0);

        // Pointer to Fock(s, k_ibz, :, :) in row-major
        const auto* F_ibz = Fock_fbz + (s * nk + sym_data.k_reduced_to_full[k_ibz]) * naosq;
        // Pointer to Fock(s, k_full, :, :) reference
        const auto* F_ref = Fock_fbz + (s * nk + k_full) * naosq;

        // GPU transform
        cudaMemcpy(d_in, F_ibz, naosq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        sym.transform_k_ao_device(handle, stream, d_in, k_full, d_out, /*nts=*/1, /*ns=*/1);
        cudaStreamSynchronize(stream);
        cudaMemcpy(result_host.data(), d_out, naosq * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        for (int ij = 0; ij < naosq; ++ij) {
          double err = std::abs(result_host[ij] - F_ref[ij]);
          if (err > max_err_gpu) {
            max_err_gpu = err;
            worst_k_gpu = k_full;
            worst_s_gpu = s;
          }
        }

        // CPU reference: U_rm * F_ibz * U_rm† with TR conj of result
        cpu_transform(U_rm + k_full * naosq, F_ibz, cpu_result.data(), nao, tr_conj);
        for (int ij = 0; ij < naosq; ++ij) {
          double err = std::abs(cpu_result[ij] - F_ref[ij]);
          if (err > max_err_cpu) {
            max_err_cpu = err;
            worst_k_cpu = k_full;
            worst_s_cpu = s;
          }
          double diff = std::abs(result_host[ij] - cpu_result[ij]);
          if (diff > max_err_gpu_vs_cpu) max_err_gpu_vs_cpu = diff;
        }
      }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    std::cout << "Symmetry transform roundtrip (Fock_fbz -> IBZ -> Fock_fbz):\n"
              << "  GPU vs ref max error: " << max_err_gpu << " at k=" << worst_k_gpu << " s=" << worst_s_gpu << "\n"
              << "  CPU vs ref max error: " << max_err_cpu << " at k=" << worst_k_cpu << " s=" << worst_s_cpu << "\n"
              << "  GPU vs CPU max error: " << max_err_gpu_vs_cpu << "\n";

    // Per-k error breakdown for diagnosis
    std::cout << "  Per-k max errors (GPU vs ref, s=0):\n";
    for (int k_full = 0; k_full < nk; ++k_full) {
      size_t k_ibz = sym_data.k_full_to_reduced[k_full];
      bool is_ibz = (sym_data.k_reduced_to_full[k_ibz] == static_cast<size_t>(k_full));
      bool tr = (sym_data.k_tr_conj[k_full] != 0);
      const auto* F_ibz = Fock_fbz + sym_data.k_reduced_to_full[k_ibz] * naosq;
      const auto* F_ref = Fock_fbz + k_full * naosq;
      cpu_transform(U_rm + k_full * naosq, F_ibz, cpu_result.data(), nao, tr);
      double kerr = 0;
      for (int ij = 0; ij < naosq; ++ij) {
        double e = std::abs(cpu_result[ij] - F_ref[ij]);
        if (e > kerr) kerr = e;
      }
      std::cout << "    k=" << k_full << " ibz=" << k_ibz
                << (is_ibz ? " (IBZ)" : "") << (tr ? " TR" : "")
                << "  err=" << kerr << "\n";
    }

    return max_err_gpu;
  }

}  // namespace green::gpu
