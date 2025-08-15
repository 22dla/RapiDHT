#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

//  вадратные
void matrixMultiplication(const double* A, const double* B, double* C, int N);
void matrixTranspose(double* A, int N);
// Ћюбые
void matrixMultiplication(const double* A, const double* B, double* C, int M, int K, int N);
void matrixTranspose(const double* A, double* B, int rows, int cols);
void vectorMatrixMultiplication(const double* A, const double* B, double* C, int N);
void BracewellTransform2D(double* A, int N);

#endif
