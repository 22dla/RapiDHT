#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

namespace RapiDHT {

//  вадратные
void MatrixMultiplication(const double* A, const double* B, double* C, int N);
void MatrixTranspose(double* A, int N);
void BracewellTransform2D(double* A, int N);

// Ћюбые
void MatrixMultiplication3D_Z(const double* d_input, const double* d_transformZ, double* d_output, int W, int H, int D);
void MatrixMultiplication(const double* A, const double* B, double* C, int M, int K, int N);
void MatrixTranspose(const double* A, double* B, int rows, int cols);
void VectorMatrixMultiplication(const double* A, const double* B, double* C, int N);
void BracewellTransform3D(double* d_A, int W, int H, int D);

} // namespace RapiDHT

#endif
