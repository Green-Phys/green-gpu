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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

/*cuDoubleComplex ADD(cuDoubleComplex x, cuDoubleComplex y);
cuComplex ADD(cuComplex x, cuComplex y);

cuDoubleComplex SUB(cuDoubleComplex x, cuDoubleComplex y);
cuComplex SUB(cuComplex x, cuComplex y);*/

cublasStatus_t ASUM(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
cublasStatus_t ASUM(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);

cublasStatus_t RSCAL(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);
cublasStatus_t RSCAL(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);

cublasStatus_t RAXPY(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);
cublasStatus_t RAXPY(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);

cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *B, int ldb,
        const cuDoubleComplex *beta,
        cuDoubleComplex *C, int ldc);
cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *B, int ldb,
        const cuComplex *beta,
        cuComplex *C, int ldc);

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
        int batchCount);
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
        int batchCount);

cublasStatus_t GEAM(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *beta,
        const cuDoubleComplex *B, int ldb,
        cuDoubleComplex *C, int ldc);
cublasStatus_t GEAM(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *beta,
        const cuComplex *B, int ldb,
        cuComplex *C, int ldc);

cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex **Aarray,
        int lda,
        int *infoArray,
        int batchSize);
cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex **Aarray,
        int lda,
        int *infoArray,
        int batchSize);

cusolverStatus_t POTRS(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuDoubleComplex *A,
        int lda,
        cuDoubleComplex *B,
        int ldb,
        int *devInfo);
cusolverStatus_t POTRS(cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        int nrhs,
        const cuComplex *A,
        int lda,
        cuComplex *B,
        int ldb,
        int *devInfo);

cublasStatus_t  GETRF(cublasHandle_t handle,
        int n,
        cuComplex *const Aarray[],
        int lda,
        int *PivotArray,
        int *infoArray,
        int batchSize);
cublasStatus_t  GETRF(cublasHandle_t handle,
        int n,
        cuDoubleComplex *const Aarray[],
        int lda,
        int *PivotArray,
        int *infoArray,
        int batchSize);

cublasStatus_t  GETRS(cublasHandle_t handle,
        cublasOperation_t trans,
        int n,
        int nrhs,
        const cuComplex *const Aarray[],
        int lda,
        const int *devIpiv,
        cuComplex *const Barray[],
        int ldb,
        int *info,
        int batchSize);
cublasStatus_t  GETRS(cublasHandle_t handle,
        cublasOperation_t trans,
        int n,
        int nrhs,
        const cuDoubleComplex *const Aarray[],
        int lda,
        const int *devIpiv,
        cuDoubleComplex *const Barray[],
        int ldb,
        int *info,
        int batchSize);
