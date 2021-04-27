/*
    Functions for making samplebased hessian
*/ 

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

#include "params.cuh"
#include "DataStructure.cuh"
#include "Matrix.cuh"
#include "dynamics.cuh"


void getInvHessian( float *Hess, SampleBasedHessian *hostHess);

__global__ void getPseduoGradient(SampleBasedHessian  *Hess, float epsilon);
__global__ void ParallelSimForPseudoGrad(SampleBasedHessian *Hess, MonteCarloMPC *sample, InputSequences *MCresult, Controller *CtrPrm, float delta, int *indices);
__global__ void getCurrentUpdateResult( SampleBasedHessian *HessInfo, float *invHess );
__global__ void copyInpSeqFromSBH( InputSequences *Output, SampleBasedHessian *HessInfo);
