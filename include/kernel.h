/*
 * Project: RapiDHT
 * File: kernel.h
 * Brief: Заголовок CUDA-обёрток и ядер для линейной алгебры и преобразования Хартли.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>

namespace RapiDHT {

// Базовые операции
template <typename T>
void MatrixMultiplication(const double* A, const double* B, double* C, int N);

template <typename T>
void MatrixTranspose(double* A, int N);

template <typename T>
void BracewellTransform2D(double* A, int N);

// Расширенные операции (с параметрами размеров)
template <typename T>
void transpose_YZ_cuda(const T* d_in, T* d_out, int W, int H, int D);

template <typename T>
void permute_ZXY_simple(const T* d_in, T* d_out, int W, int H, int D);

template <typename T>
void MatrixMultiplication3D_Z(const T* d_input, const T* d_transformZ, T* d_output, int W, int H, int D);

template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N);

template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols);

template<typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N);

template <typename T>
void BracewellTransform3D(T* d_A, int W, int H, int D);

void InitializeHartleyMatrix(double* dKernel, size_t height);
void InitializeHartleyMatrix(float* dKernel, size_t height);

} // namespace RapiDHT

#endif // KERNEL_H
