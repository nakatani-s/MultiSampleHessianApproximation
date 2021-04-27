/*
params.cuh
*/ 

#ifndef PARAMS_CUH
#define PARAMS_CUH

const int DIM_OF_STATE = 4;
const int DIM_OF_U = 1;
const int NUM_OF_CONSTRAINTS = 4;
const int DIM_OF_WEIGHT_MATRIX = 5;
#ifdef COLLISION
const int NUM_OF_PARAMS = 8; 
#else
const int NUM_OF_PARAMS = 7;
#endif

// GPU parameters
const int NUM_OF_SAMPLES = 5000;
const int NUM_OF_ELITESAMPLE = 17;
const int THREAD_PER_BLOCKS = 100;


// MPC parameters
const int TIME = 100;
const int HORIZON = 100;
const int NUM_OF_RECALC = 5;
const float interval = 0.01;
const float invBarrier = 5000;
const float zeta = 0.05f;


const float initVar = 1.0f;

#endif