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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cusolverDn.h>
#include <green/gpu/cublas_routines_prec.h>

/*cuDoubleComplex ADD(cuDoubleComplex x, cuDoubleComplex y) {
  return cuCadd(x, y);
}
cuComplex ADD(cuComplex x, cuComplex y) {
  return cuCaddf(x, y);
}

cuDoubleComplex SUB(cuDoubleComplex x, cuDoubleComplex y) {
  return cuCsub(x, y);
}
cuComplex SUB(cuComplex x, cuComplex y) {
  return cuCsubf(x, y);
}*/

cublasStatus_t ASUM(cublasHandle_t handle, int n,
          const cuDoubleComplex *x, int incx, double *result) {
  return cublasDzasum(handle, n, x, incx, result);
}
cublasStatus_t ASUM(cublasHandle_t handle, int n,
          const cuComplex *x, int incx, float *result) {
  return cublasScasum(handle, n, x, incx, result);
}


cublasStatus_t RSCAL(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
  return cublasDscal(handle, n, alpha, x, incx);
}
cublasStatus_t RSCAL(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
  return cublasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t RAXPY(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
cublasStatus_t RAXPY(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *B, int ldb,
        const cuDoubleComplex *beta,
        cuDoubleComplex *C, int ldc) {
  return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *B, int ldb,
        const cuComplex *beta,
        cuComplex *C, int ldc) {
  return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        long long int          strideA,
        const cuDoubleComplex *B, int ldb,
        long long int          strideB,
        const cuDoubleComplex *beta,
        cuDoubleComplex       *C, int ldc,
        long long int          strideC,
        int batchCount) {
  return cublasZgemmStridedBatched(handle, transa, transb, m, n, k,
          alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        long long int          strideA,
        const cuComplex *B, int ldb,
        long long int          strideB,
        const cuComplex *beta,
        cuComplex       *C, int ldc,
        long long int          strideC,
        int batchCount) {
  return cublasCgemmStridedBatched(handle, transa, transb, m, n, k,
          alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t GEAM(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *beta,
        const cuDoubleComplex *B, int ldb,
        cuDoubleComplex *C, int ldc) {
  return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
cublasStatus_t GEAM(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *beta,
        const cuComplex *B, int ldb,
        cuComplex *C, int ldc) {
  return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex **Aarray,
        int lda,
        int *infoArray,
        int batchSize) {
  return cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}
cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex **Aarray,
        int lda,
        int *infoArray,
        int batchSize) {
  return cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t POTRS(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuDoubleComplex *A,
        int lda,
        cuDoubleComplex *B,
        int ldb,
        int *devInfo) {
  return cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
cusolverStatus_t POTRS(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuComplex *A,
        int lda,
        cuComplex *B,
        int ldb,
        int *devInfo) {
  return cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
