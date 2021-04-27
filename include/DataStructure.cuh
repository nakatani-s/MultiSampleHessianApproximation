#include <curand_kernel.h>
#include "params.cuh"
#ifndef DATASTRUCTUE_CUH
#define DATASTRUCTUE_CUH

typedef struct{
    float L;
    float W;
    float InputSeq[DIM_OF_U][HORIZON];
}MonteCarloMPC;

typedef struct{
    float State[DIM_OF_STATE];
    float Param[NUM_OF_PARAMS];
    float Constraints[NUM_OF_CONSTRAINTS];
    float WeightMatrix[DIM_OF_WEIGHT_MATRIX];
}Controller;

typedef struct{
    float currentU[DIM_OF_U][HORIZON];
    float pseudoGradient[HORIZON];
    float cost[HORIZON][3];

    float modified_U[DIM_OF_U][HORIZON];
    float delta_u[DIM_OF_U][HORIZON];

}SampleBasedHessian;


typedef struct{
    float InputSeq[DIM_OF_U];
}InputSequences;
#endif

__global__ void setupMonteCarloMPCVectors(MonteCarloMPC *Out);