/*
    Functions for making samplebased hessian
*/
#include "../include/SampleBasedHessian.cuh"

__global__ void makeGrad_SamplePointMatrix(float *G, float *SPM, SampleBasedHessian *Subject)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    G[id] = Subject[0].pseudoGradient[threadIdx.x] - Subject[blockIdx.x + 1].pseudoGradient[threadIdx.x];
    SPM[id] = Subject[0].currentU[0][threadIdx.x] - Subject[blockIdx.x + 1].currentU[0][threadIdx.x];
    __syncthreads();
}

void getInvHessian( float *Hess, SampleBasedHessian *hostHess)
{
    // cublasHandle_t handle_invHess;
    // CUBLAS_CHECK( cublasCreate(&cublas_status), "Failed to initialize cuBLAS");
    SampleBasedHessian *deviceSubject;
    float *deviceArrayHess;
    float *arrayGrad, *device_arrayGrad;
    float *invArrayGrad, *deviceInvArrayGrad;
    float /* *arrayVect, */ *device_arrayVect;

    size_t szMat = HORIZON * HORIZON * sizeof(float);

    int nx = HORIZON;
    int ny = HORIZON;
    dim3 block(1,1);
    dim3 grid(( nx  + block.x - 1)/ block.x, ( ny + block.y -1) / block.y);

    arrayGrad = (float *)malloc(HORIZON * HORIZON * sizeof(float));
    CHECK_CUDA( cudaMalloc( &deviceArrayHess, szMat), "Failed to allocate array Hessian on device matrix");
    CHECK_CUDA( cudaMalloc( &device_arrayGrad,  szMat), "Failed to allocate array Grad on device matrix" );
    CHECK_CUDA( cudaMalloc( &device_arrayVect, szMat), "Failed to allocate array Vect on device matrix");
    CHECK_CUDA( cudaMalloc( &deviceInvArrayGrad, szMat), "Failed to allocate array invGrad on device matrix");
    CHECK_CUDA( cudaMalloc( &deviceSubject, (HORIZON + 1) * sizeof(SampleBasedHessian)), "Failed to allocate SampledBasedHessianVector in SBH.cu");
    CHECK_CUDA( cudaMemcpy(deviceSubject, hostHess, (HORIZON + 1) * sizeof(SampleBasedHessian), cudaMemcpyHostToDevice), "Failed to copy SampledBasedHessianVector in SBH.cu");
    invArrayGrad = (float *)malloc( szMat );
    // arrayVect = (float *)malloc( szMat );

    makeGrad_SamplePointMatrix<<<HORIZON, HORIZON>>>( device_arrayGrad, device_arrayVect, deviceSubject );
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");
    CHECK_CUDA( cudaMemcpy( arrayGrad, device_arrayGrad, szMat, cudaMemcpyDeviceToHost), "Failed to copy matrix G in SBH.cu");
    // CHECK_CUDA( cudaMemcpy( arrayVect, device_arrayVect, szMat, cudaMemcpyDeviceToHost), "Failed to copy matrix V in SBH.cu");

    GetInvMatrix(invArrayGrad, arrayGrad, HORIZON);
    CHECK_CUDA( cudaMemcpy(deviceInvArrayGrad, invArrayGrad, szMat, cudaMemcpyHostToDevice), "Failed to copy inverse matrix G to device");

    GetResultMatrixProduct<<<grid, block>>>( deviceArrayHess, device_arrayVect, deviceInvArrayGrad, HORIZON );
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUDA( cudaMemcpy(Hess, deviceArrayHess, szMat, cudaMemcpyDeviceToHost), "Failed to copy inverse Hess to Host");

}

__device__ void readParam(float *prm, Controller CtrPrm){
    for(int i = 0; i < NUM_OF_PARAMS; i++)
    {
        prm[i] = CtrPrm.Param[i];
    }
}

__global__ void getPseduoGradient(SampleBasedHessian  *Hess, float epsilon)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float temp[2] = { };

    temp[0] = Hess[blockIdx.x].cost[threadIdx.x][1] - Hess[blockIdx.x].cost[threadIdx.x][2];
    temp[1] = 2 * epsilon;
    Hess[blockIdx.x].pseudoGradient[threadIdx.x] = temp[0] / temp[1];
    __syncthreads();
}


__global__ void getCurrentUpdateResult( SampleBasedHessian *HessInfo, float *invHess )
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

    float delta_u = 0.0f;
    float result_u = 0.0f;

    // 以上なidの時は処理しないおまじない
    if(threadIdx.x < HORIZON && blockIdx.x < HORIZON)
    {
        for(int i = 0; i < HORIZON; i++)
        {
            delta_u += invHess[ blockIdx.x * HORIZON + i] * HessInfo[threadIdx.x].pseudoGradient[i];
        }
        result_u = HessInfo[threadIdx.x].currentU[0][blockIdx.x] - delta_u; // current_u - H^-1*g = tilde{u}^{*}
        HessInfo[threadIdx.x].modified_U[0][blockIdx.x] = result_u;
        HessInfo[threadIdx.x].delta_u[0][blockIdx.x] = delta_u;
    }
    __syncthreads();
}

__global__ void copyInpSeqFromSBH( InputSequences *Output, SampleBasedHessian *HessInfo)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id < HORIZON)
    {
        Output[id].InputSeq[0] = HessInfo[0].modified_U[0][id];
    }
    __syncthreads();
}

__global__ void ParallelSimForPseudoGrad(SampleBasedHessian *Hess, MonteCarloMPC *sample, InputSequences *MCresult, Controller CtrPrm, float delta, int *indices)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int id = iy * 3 + ix;
    unsigned int id =  iy * 3 * (HORIZON + 1) + ix;

    float stageCost = 0.0f;
    float totalCost = 0.0f;
    InputSequences *InputSeqInThread;
    InputSeqInThread = (InputSequences *)malloc(sizeof(InputSeqInThread) * HORIZON);
    float stateInThisThreads[DIM_OF_STATE] = { };
    float dstateInThisThreads[DIM_OF_STATE] = { };

    float d_param[NUM_OF_PARAMS];
    readParam(d_param, CtrPrm);

    for(int i = 0; i < DIM_OF_STATE; i++){
        stateInThisThreads[i] = CtrPrm.State[i];
    }

    for(int t = 0; t < HORIZON; t++){
        for(int uIndex = 0; uIndex < DIM_OF_U; uIndex++ )
        {
            if(blockIdx.x < 1 ){
                if( t == iy)
                {
                    if(threadIdx.x == 0)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex];
                    }
                    if(threadIdx.x == 1)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex] + delta;
                    }
                    if(threadIdx.x == 2)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex] - delta;
                    }
                }else{
                    InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex];
                }
            }else{
                if( t == iy)
                {
                    if(threadIdx.x == 0)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t];
                    }
                    if(threadIdx.x == 1)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t] + delta;
                    }
                    if(threadIdx.x == 2)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t] - delta;
                    }
                }else{
                    InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t];
                }

            }
        }
        for(int sec = 0; sec < 1; sec++){
            dstateInThisThreads[0] = stateInThisThreads[2];
            dstateInThisThreads[1] = stateInThisThreads[3];
                /*
                dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
                dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
                */
            dstateInThisThreads[2] = Cart_type_Pendulum_ddx(InputSeqInThread[t].InputSeq[0], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
            dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(InputSeqInThread[t].InputSeq[0], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
            stateInThisThreads[2] = stateInThisThreads[2] + (interval * dstateInThisThreads[2]);
            stateInThisThreads[3] = stateInThisThreads[3] + (interval * dstateInThisThreads[3]);
            stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
            stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);
#ifdef COLLISION
            if(stateInThisThreads[0] <= CtrPrm.Constraints[2]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = CtrPrm.Constraints[2];
            }
            if(CtrPrm.Constraints[3] <= stateInThisThreads[0]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = CtrPrm.Constraints[3];
            }
#endif
        }
        while(stateInThisThreads[1] > M_PI)
            stateInThisThreads[1] -= (2 * M_PI);
        while(stateInThisThreads[1] < -M_PI)
            stateInThisThreads[1] += (2 * M_PI);
            
        stageCost = stateInThisThreads[0] * stateInThisThreads[0] * CtrPrm.WeightMatrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * CtrPrm.WeightMatrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * CtrPrm.WeightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * CtrPrm.WeightMatrix[3]
            + InputSeqInThread[t].InputSeq[0] * InputSeqInThread[t].InputSeq[0] * CtrPrm.WeightMatrix[4];

#ifndef COLLISION
        if(stateInThisThreads[0] <= 0){
            stageCost += 1 / (powf(stateInThisThreads[0] - CtrPrm.Constraints[2],2) * invBarrier);
            if(stateInThisThreads[0] < CtrPrm.Constraints[2]){
                stageCost += 1000000;
            }
        }else{
            stageCost += 1 / (powf(CtrPrm.Constraints[3] - stateInThisThreads[0],2) * invBarrier);
            if(stateInThisThreads[0] > CtrPrm.Constraints[3]){
                stageCost += 1000000;
            }
        }
#endif
        totalCost += stageCost;

        stageCost = 0.0f;
    }
    Hess[blockIdx.x].currentU[0][blockIdx.y] =  
    Hess[blockIdx.x].cost[blockIdx.y][threadIdx.x] = totalCost;
    __syncthreads();
}