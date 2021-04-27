/*
    Functions for matrix operations
*/
#include "../include/Matrix.cuh"

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}


void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}

void shift_Input_vec( InputSequences *inputVector, int uIndex)
{
    float temp[HORIZON]= { };
    for(int i = 0; i < HORIZON - 1; i++){
        temp[i] = inputVector[i+1].InputSeq[uIndex];
    }
    temp[HORIZON - 1] = inputVector[HORIZON - 1].InputSeq[uIndex];
    for(int i = 0; i < HORIZON; i++){
        inputVector[i].InputSeq[uIndex] = temp[i];
    }
}

void GetInvMatrix(float *invMat, float *originMat, int num)
{
    cublasHandle_t cublas_status;
    CHECK_CUBLAS(cublasCreate(&cublas_status),"Failed to initialize cuBLAS");

    float **arrayA;
    float **arrayC;
    float *d_arrayA;
    float *d_arrayC;
    int *d_LUPivots;
    int *d_LUInfo;

    size_t szMat = num * num * sizeof(float);

    CHECK_CUDA(cudaMalloc(&arrayA, sizeof(float*)), "Failed to allocate arrayA");
    CHECK_CUDA(cudaMalloc(&arrayC, sizeof(float*)), "Failed to allocate arrayC");
    CHECK_CUDA(cudaMalloc(&d_arrayA, szMat), "Failed to allocate d_arrayA");
    CHECK_CUDA(cudaMalloc(&d_arrayC, szMat), "Failed to allocate d_arrayC");
    CHECK_CUDA(cudaMalloc(&d_LUPivots, sizeof(int)), "Failed to allocate arrayC");
    CHECK_CUDA(cudaMalloc(&d_LUInfo, sizeof(int)), "Failed to allocate arrayC");

    CHECK_CUDA(cudaMemcpy(d_arrayA, originMat, szMat, cudaMemcpyHostToDevice), "Failed to copy Origin Matrix to d_arrayA");
    CHECK_CUDA(cudaMemcpy(arrayA, &d_arrayA, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to arrayA");
    CHECK_CUDA(cudaMemcpy(arrayC, &d_arrayC, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to arrayC");

    CHECK_CUBLAS(cublasSgetrfBatched(cublas_status, num, arrayA, num, d_LUPivots, d_LUInfo, 1), "Failed to perform LU decomp operation");
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUBLAS(cublasSgetriBatched(cublas_status, num, (const float **)arrayA, num, d_LUPivots, arrayC, num, d_LUInfo, 1), "Failed to perform Inverse operation!");
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUDA(cudaMemcpy(invMat, d_arrayC, szMat, cudaMemcpyDeviceToHost), "Failed to copy to invMat");

    CHECK_CUDA(cudaFree(arrayA),"Failed to free arrayA");
    CHECK_CUDA(cudaFree(arrayC),"Failed to free arrayC");
    CHECK_CUDA(cudaFree(d_arrayA),"Failed to free d_arrayA");
    CHECK_CUDA(cudaFree(d_arrayC),"Failed to free d_arrayC");
    CHECK_CUDA(cudaFree(d_LUPivots),"Failed to free d_LUPivots");
    CHECK_CUDA(cudaFree(d_LUInfo),"Failed to free d_LUInfo");

    CHECK_CUBLAS(cublasDestroy(cublas_status), "Failed to destory cuBLAS");

}

__global__ void GetResultMatrixProduct( float *ans, float *lmat, float *rmat, const int nx )
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * nx  + ix;

    if( ix < HORIZON && iy < HORIZON)
    {
        float el = 0.0f;
        for(int i = 0; i < HORIZON; i++)
        {
            el += lmat[ iy * nx + i] * rmat[ i * nx + ix ];
        }
        ans[id] = el;
    }
    __syncthreads();
}

