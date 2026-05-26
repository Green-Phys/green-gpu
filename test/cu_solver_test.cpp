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

#include <green/gpu/gpu_factory.h>
#include <green/grids.h>

#include "catch2/matchers/catch_matchers.hpp"
#include "cu_symmetry_test.h"
#include "tensor_test.h"


#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>


void solve_hf(const std::string& input, const std::string& int_hf, const std::string& data, const std::string& mem) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_hf;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_DAMPING --damping 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_hf_file=" + df_int_path +
      " --cuda_low_gpu_memory " + mem + " --cuda_low_cpu_memory " + mem + " --verbose=5";
  green::grids::define_parameters(p);
  green::gpu::custom_kernel_parameters(p);
  green::symmetry::define_parameters(p);
  p.define<std::string>("dfintegral_hf_file", "Path to Hartree-Fock integrals");
  p.define<std::string>("dfintegral_file", "Path to integrals for high orfer theories");
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  size_t                                nao, nso, NQ, ns, nk, ink, nts;
  double                                madelung;
  green::gpu::ztensor<4>                tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nao"] >> nao;
    ar["params/nso"] >> nso;
    ar["params/NQ"] >> NQ;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["symmetry/k/ink"] >> ink;
    ar["HF/madelung"] >> madelung;
    green::gpu::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared    = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared    = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1      = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sigma1_test = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sk          = green::gpu::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context().node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    ar["result/Sigma1"] >> Sigma1_test;
    ar.close();
  }
  double prefactor = (ns == 2 or nao != nso) ? -1.0 : -2.0;
  green::gpu::ztensor<4> dm(ns, ink, nso, nso);
  dm << G_shared.object()(G_shared.object().shape()[0] - 1);
  auto [kernel, solver] = green::gpu::custom_hf_kernel(nso != nao, p, nao, nso, ns, NQ, madelung, bz, Sk);
  Sigma1 << solver(dm);
  Sigma1 *= prefactor;
  REQUIRE_THAT(Sigma1, IsCloseTo(Sigma1_test, 1e-8));
}


// Check that one step of scf_type ("HF" or "GW") gives the same Sigma at every
// IBZ k-point regardless of whether space-group, TR-only, or no symmetry is used.
// Uses a cubic Ar (def2-svp, --x2c 2, 3x3x1 k-mesh) reference that exercises
// both the orbital point-group representation (the basis includes p-shells)
// and the SU(2) spinor part of the double-group transform under full X2C
// spin-orbit coupling.  The 3x3x1 mesh gives non-trivial TR reduction
// (no_symm=9 ink, trs_only=5 ink, full_symm=3 ink).  Tolerance accommodates
// the integral-storage-floor disagreement between symmetry-reduced and
// no-symmetry runs.
void check_x2c_ar_symmetry(const std::string& scf_type, const std::string& lin, const std::string& mem) {
  const std::string dir       = TEST_PATH + "/GW_X2C_Ar"s;
  // Shared integral dir for both HF and GW (saves test data footprint).
  const std::string df_path   = dir + "/df_hf_int"s;
  const std::string grid_file = GRID_PATH + "/ir/1e4.h5"s;
  constexpr double  tol = 1e-5;

  size_t nts = 0;
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    nts += 2;
  }

  // Run one SCF step and return Sigma: [nts,ns,ink,nso,nso] for GW,
  // [1,ns,ink,nso,nso] for HF (Sigma1 broadcast to uniform shape).
  // All dimensions are read from input_file to avoid brittle hardcoding.
  auto run = [&](const std::string& input_file, const std::string& data_file) {
    size_t nao, nso, ns, nk, ink, NQ;
    double madelung;
    {
      green::h5pp::archive ar(input_file);
      ar["params/nao"]      >> nao;
      ar["params/nso"]      >> nso;
      ar["params/ns"]       >> ns;
      ar["params/nk"]       >> nk;
      ar["params/NQ"]       >> NQ;
      ar["symmetry/k/ink"]  >> ink;
      ar["HF/madelung"]     >> madelung;
    }

    auto        p    = green::params::params("DESCR");
    std::string args = "test --restart 0 --itermax 1 --E_thr 1e-13 "
                       "--mixing_type SIGMA_DAMPING --damping 0.7 "
                       "--input_file=" + input_file + " --BETA 10 --grid_file=" + grid_file +
                       " --cuda_low_gpu_memory " + mem + " --cuda_low_cpu_memory " + mem +
                       (scf_type == "GW" ? " --dfintegral_file=" + df_path + " --cuda_linear_solver=" + lin
                                         : " --dfintegral_hf_file=" + df_path);
    green::grids::define_parameters(p);
    green::gpu::custom_kernel_parameters(p);
    green::symmetry::define_parameters(p);
    p.define<std::string>("dfintegral_hf_file", "Path to HF integrals");
    p.define<std::string>("dfintegral_file", "Path to integrals for correlated methods");
    p.define<bool>("P_sp", "Compute polarization in single precision", false);
    p.define<bool>("Sigma_sp", "Compute self-energy in single precision", false);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);

    green::gpu::ztensor<4> tmp(ns, nk, nso, nso), Sigma1(ns, ink, nso, nso), Sk(ns, ink, nso, nso);
    {
      green::h5pp::archive    ar(input_file);
      green::gpu::dtensor<5>  S_k;
      ar["HF/S-k"] >> S_k;
      tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
    }
    for (size_t is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));

    auto G_shared = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
    auto S_shared = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
    {
      green::h5pp::archive ar(data_file, "r");
      G_shared.fence();
      if (!green::utils::context().node_rank) ar["G_tau"] >> G_shared.object();
      G_shared.fence();
    }

    size_t                 nt_out = nts;
    green::gpu::ztensor<5> result;
    if (scf_type == "GW") {
      green::grids::transformer_t ft(p);
      auto [kernel, solver] = green::gpu::custom_gw_kernel(nso != nao, p, nao, nso, ns, NQ, ft, bz, Sk);
      solver(G_shared, S_shared);
      result.resize(nts, ns, ink, nso, nso);
      S_shared.fence();
      if (!green::utils::context().node_rank) result << S_shared.object();
      S_shared.fence();
    } else {
      auto [kernel, solver] = green::gpu::custom_hf_kernel(nso != nao, p, nao, nso, ns, NQ, madelung, bz, Sk);
      green::gpu::ztensor<4> dm(ns, ink, nso, nso);
      dm << G_shared.object()(G_shared.object().shape()[0] - 1);
      Sigma1 << solver(dm);
      double prefactor = (ns == 2 or nao != nso) ? -1.0 : -2.0;
      Sigma1 *= prefactor;
      result.resize(1, ns, ink, nso, nso);
      if (!green::utils::context().node_rank) result(0) << Sigma1;
      nt_out = 1;
    }
    // Broadcast result to all ranks so assertions run on consistent data.
    MPI_Bcast(result.data(), result.size(), MPI_CXX_DOUBLE_COMPLEX, 0, green::utils::context().global);
    return std::make_pair(result, nt_out);
  };

  auto [Sigma_nosymm, nts_out] = run(dir + "/input_no_symm.h5"s,   dir + "/data_no_symm.h5"s);
  auto [Sigma_symm,   _1]      = run(dir + "/input_full_symm.h5"s, dir + "/data_full_symm.h5"s);
  auto [Sigma_trs,    _2]      = run(dir + "/input_trs_only.h5"s,  dir + "/data_trs_only.h5"s);

  std::vector<long> ibz2bz_symm, ibz2bz_trs;
  {
    green::h5pp::archive ar(dir + "/input_full_symm.h5"s, "r");
    ar["symmetry/k/ibz2bz"] >> ibz2bz_symm;
  }
  {
    green::h5pp::archive ar(dir + "/input_trs_only.h5"s, "r");
    ar["symmetry/k/ibz2bz"] >> ibz2bz_trs;
  }

  auto check = [&](const green::gpu::ztensor<5>& Sigma_ref, const green::gpu::ztensor<5>& Sigma_sym,
                   const std::vector<long>& ibz2bz, const char* label) {
    const size_t nso_ck = Sigma_sym.shape()[3];
    const size_t nao_ck = nso_ck / 2;  // X2C: nso = 2 * nao
    for (size_t i = 0; i < ibz2bz.size(); ++i) {
      size_t k = ibz2bz[i];
      for (size_t t = 0; t < nts_out; ++t) {
        // Diagnostic: if any element of this (i,t) slice exceeds tol, dump
        // per-spinor-block max diff before REQUIRE_THAT throws.
        double overall_max = 0;
        for (size_t r = 0; r < nso_ck; ++r)
          for (size_t c = 0; c < nso_ck; ++c)
            overall_max = std::max(overall_max,
                std::abs(Sigma_sym(t, 0, i, r, c) - Sigma_ref(t, 0, k, r, c)));
        if (overall_max > tol) {
          auto block_max = [&](size_t r0, size_t c0, const char* name) {
            double m = 0; size_t mr = 0, mc = 0;
            for (size_t r = 0; r < nao_ck; ++r)
              for (size_t c = 0; c < nao_ck; ++c) {
                double d = std::abs(Sigma_sym(t, 0, i, r0 + r, c0 + c) -
                                    Sigma_ref(t, 0, k, r0 + r, c0 + c));
                if (d > m) { m = d; mr = r; mc = c; }
              }
            std::cout << "      " << name << " max=" << m
                      << " at (" << mr << "," << mc << ")" << std::endl;
          };
          std::cout << "  [" << label << "] i=" << i << " k=" << k << " t=" << t
                    << " overall max=" << overall_max << " (tol=" << tol << ")"
                    << std::endl;
          block_max(0,      0,      "aa");
          block_max(nao_ck, nao_ck, "bb");
          block_max(0,      nao_ck, "ab");
          block_max(nao_ck, 0,      "ba");
        }
        REQUIRE_THAT(Sigma_sym(t, 0, i), IsCloseTo(Sigma_ref(t, 0, k), tol));
      }
    }
  };
  check(Sigma_nosymm, Sigma_symm, ibz2bz_symm, "full_symm vs no_symm");
  check(Sigma_nosymm, Sigma_trs,  ibz2bz_trs,  "trs_only vs no_symm");
}

void solve_gw(const std::string& input, const std::string& int_f, const std::string& data, const std::string& lin, const std::string& mem, bool sp, const std::string& nt_batch) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_f;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_DAMPING --damping 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path + " --verbose=5 " +
      " --cuda_low_gpu_memory " + mem + " --cuda_low_cpu_memory " + mem + " --cuda_linear_solver=" + lin + " --nt_batch=" + nt_batch;
  green::grids::define_parameters(p);
  green::symmetry::define_parameters(p);
  green::gpu::custom_kernel_parameters(p);
  p.define<std::string>("dfintegral_hf_file", "Path to Hartree-Fock integrals");
  p.define<std::string>("dfintegral_file", "Path to integrals for high orfer theories");
  p.define<bool>("P_sp", "Compute polarization in single precision", sp);
  p.define<bool>("Sigma_sp", "Compute self-energy in single precision", sp);
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  green::grids::transformer_t           ft(p);
  size_t                                NQ, nao, nso, ns, nk, ink, nts;
  green::gpu::ztensor<4>                tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nso"] >> nso;
    ar["params/nao"] >> nao;
    ar["params/NQ"] >> NQ;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["symmetry/k/ink"] >> ink;
    green::gpu::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared     = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared     = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared_tst = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1       = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sk           = green::gpu::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context().node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    S_shared_tst.fence();
    if (!green::utils::context().node_rank) ar["result/Sigma_tau"] >> S_shared_tst.object();
    S_shared_tst.fence();
    ar.close();
  }

  auto [kernel, solver] = green::gpu::custom_gw_kernel(nso != nao, p, nao, nso, ns, NQ, ft, bz, Sk);
  solver(G_shared, S_shared);
  REQUIRE_THAT(S_shared.object(), IsCloseTo(S_shared_tst.object(), 1e-6));
}

TEST_CASE("GPU Solver") {
  SECTION("GW_LU") {
    // solve_gw(input_file, df_int_path, test_file, linear_solver, low_mem, sp_precision, nt_batch);
    // automatically optimize nt_batch
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "false", false, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "true", false, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "false", true, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "true", true, "0");
    // set nt_batch = 1
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "false", false, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "true", false, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "false", true, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU", "true", true, "1");
  }
  SECTION("GW_Cholesky") {
    // solve_gw(input_file, df_int_path, test_file, linear_solver, low_mem, sp_precision, nt_batch);
    // automatically optimize nt_batch
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "false", false, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "false", true, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "true", false, "0");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "true", true, "0");
    // set nt_batch = 1
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "false", false, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "false", true, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "true", false, "1");
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky", "true", true, "1");
  }
  SECTION("GW_X2C") {
    // solve_gw(input_file, df_int_path, test_file, linear_solver, low_mem, sp_precision, nt_batch);
    // automatically optimize nt_batch
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "false", false, "0");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "false", true, "0");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "true", false, "0");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "true", true, "0");
    // set nt_batch = 1
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "false", false, "1");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "false", true, "1");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "true", false, "1");
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU", "true", true, "1");
  }

  SECTION("HF") {
    solve_hf("/HF/input.h5", "/HF/df_hf_int", "/HF/data.h5", "false");
    solve_hf("/HF/input.h5", "/HF/df_hf_int", "/HF/data.h5", "true");
  }
  SECTION("HF_X2C") {
    solve_hf("/HF_X2C/input.h5", "/HF_X2C/df_hf_int", "/HF_X2C/data.h5", "false");
    solve_hf("/HF_X2C/input.h5", "/HF_X2C/df_hf_int", "/HF_X2C/data.h5", "true");
  }

  SECTION("HF_X2C_Ar_Symmetry") { check_x2c_ar_symmetry("HF", "LU", "false"); }
  SECTION("GW_X2C_Ar_Symmetry") { check_x2c_ar_symmetry("GW", "LU", "false"); }

  SECTION("Symmetry_Transform") {
    std::string input_file = TEST_PATH + "/GW/input.h5"s;
    auto p = green::params::params("DESCR");
    std::string args = "test --input_file=" + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);

    size_t nao, ns, nk;
    green::gpu::dtensor<5> Fock_raw;
    {
      green::h5pp::archive ar(input_file);
      ar["params/nao"] >> nao;
      ar["params/ns"] >> ns;
      ar["params/nk"] >> nk;
      ar["HF/Fock-k"] >> Fock_raw;
      ar.close();
    }
    // Fock_raw is dtensor<5> (ns, nk, nao, nao, 2) with __complex__ attribute
    green::gpu::ztensor<4> Fock_fbz(ns, nk, nao, nao);
    Fock_fbz << Fock_raw.view<std::complex<double>>().reshape(ns, nk, nao, nao);

    // Build cu_symmetry_data (same as make_cu_symmetry_data in gw_gpu_kernel.cpp)
    green::gpu::cu_symmetry_data sym_data;
    const auto& ksym  = bz.k_symmetry();
    const auto& qsym  = bz.q_symmetry();
    const auto& kqmap = bz.k_q_map();

    sym_data.k_full_to_reduced = ksym.full_to_reduced();
    sym_data.k_reduced_to_full = ksym.reduced_to_full();
    sym_data.k_tr_conj         = ksym.tr_conj_list();
    sym_data.nk  = sym_data.k_full_to_reduced.size();
    sym_data.ink = sym_data.k_reduced_to_full.size();
    sym_data.k_stars.resize(sym_data.ink);
    for (size_t ik = 0; ik < sym_data.ink; ++ik) sym_data.k_stars[ik] = ksym.star(ik);

    sym_data.q_full_to_reduced = qsym.full_to_reduced();
    sym_data.q_reduced_to_full = qsym.reduced_to_full();
    sym_data.q_tr_conj         = qsym.tr_conj_list();
    sym_data.nq  = sym_data.q_full_to_reduced.size();
    sym_data.inq = sym_data.q_reduced_to_full.size();
    sym_data.q_stars.resize(sym_data.inq);
    for (size_t iq = 0; iq < sym_data.inq; ++iq) sym_data.q_stars[iq] = qsym.star(iq);

    sym_data.k1_from_k2q_map.resize(sym_data.nk * sym_data.nq);
    sym_data.k2_from_k1q_map.resize(sym_data.nk * sym_data.nq);
    for (size_t k = 0; k < sym_data.nk; ++k)
      for (size_t q = 0; q < sym_data.nq; ++q) {
        sym_data.k1_from_k2q_map[k * sym_data.nq + q] = kqmap.k1_from_k2q(k, q);
        sym_data.k2_from_k1q_map[k * sym_data.nq + q] = kqmap.k2_from_k1q(k, q);
      }

    sym_data.k_ao_transforms.resize(sym_data.nk * nao * nao);
    green::symmetry::MatrixXcd U_k(nao, nao);
    for (size_t k = 0; k < sym_data.nk; ++k) {
      ksym.k_sym_transform_ao(U_k, k);
      std::memcpy(sym_data.k_ao_transforms.data() + k * nao * nao, U_k.data(), nao * nao * sizeof(std::complex<double>));
    }

    double max_err = green::gpu::test_symmetry_transform_roundtrip(
        sym_data, Fock_fbz.data(),
        static_cast<int>(ns), static_cast<int>(nk), static_cast<int>(nao));
    REQUIRE(max_err < 1e-4);
    // TODO: The max_err tol is very large (at 1e-4). But this is because DFT Fock input is 
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
