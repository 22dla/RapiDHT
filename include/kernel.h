#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

namespace RapiDHT {

//  вадратные
void MatrixMultiplication(const double* A, const double* B, double* C, int N);
void MatrixTranspose(double* A, int N);
void BracewellTransform2D(double* A, int N);

// Ћюбые
void MatrixMultiplication(const double* A, const double* B, double* C, int M, int K, int N);
void MatrixTranspose(const double* A, double* B, int rows, int cols);
void VectorMatrixMultiplication(const double* A, const double* B, double* C, int N);

} // namespace RapiDHT

#endif
