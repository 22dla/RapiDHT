#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

void matrixMultiplication(double *A, double *B, double *C, int N);
void vectorMatrixMultiplication(double* A, double* B, double* C, int N);
void matrixTranspose(double* A, int N);
void BracewellTransform2D(double* A, int N);

#endif
