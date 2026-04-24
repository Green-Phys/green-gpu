#ifndef CU_SYMMETRY_TEST_H
#define CU_SYMMETRY_TEST_H

#include <green/gpu/cu_symmetry.h>
#include <complex>
#include <string>

namespace green::gpu {
  std::string test_symmetry_transform_roundtrip(
      const cu_symmetry_data& sym_data,
      const std::complex<double>* Fock_fbz,  // [ns, nk, nao, nao]
      int ns, int nk, int nao);
}

#endif
