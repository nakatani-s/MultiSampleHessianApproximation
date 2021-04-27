/*
    Functions for matrix operations
    #using cuda function
    #using cuBLAS
    #using cuSOLVER
*/
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

#include "params.cuh"
#include "DataStructure.cuh"

#define CHECK_CUDA(call,str)                                                         \
{                                                                                    \
    const cudaError_t error = call;                                                  \
    if (error != cudaSuccess)                                                        \
    {                                                                                \
        printf("Cuda Error: %s : %s: %d, ", str, __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                                       \
                cudaGetErrorString(error));                                          \
        exit(1);                                                                     \
    }                                                                                \
}                                                                                   

#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}     

#ifndef MATRIX_CUH
#define MATRIX_CUH
// Function for setting GPU parameters
unsigned int countBlocks(unsigned int a, unsigned int b);

void printMatrix(int m, int n, float*A, int lda, const char* name);
void GetInvMatrix(float *invMat, float *originMat, int num);
void shift_Input_vec( InputSequences *inputVector, int uIndex);
__global__ void GetResultMatrixProduct( float *ans, float *lmat, float *rmat, const int nx ); 

#endif