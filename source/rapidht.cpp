#include <omp.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <rapidht.h>
#include <utilities.h>
#include <complex>
#include <kernel.h>
#ifdef USE_CUDA
#include <cublas_v2.h>
#endif

namespace RapiDHT {

HartleyTransform::HartleyTransform(size_t width, size_t height = 0, size_t depth = 0, Modes mode = Modes::CPU) :_mode(mode) {
	PROFILE_FUNCTION();

	if (width == 0) {
		throw std::invalid_argument("Width must be positive.");
	}
	if (height == 0 && depth > 0) {
		throw std::invalid_argument("If height is zero, depth must also be zero.");
	}

	_dims = { width, height, depth };

	// Preparation to 1D transforms
	if (_mode == Modes::CPU || _mode == Modes::RFFT) {
		for (size_t i = 0; i < _bitReversedIndices.size(); ++i) {
			_bitReversedIndices[i].resize(_dims[i]);
			BitReverse(_bitReversedIndices[i]);
		}
	}
	if (_mode == Modes::GPU) {
		// Initialize transform matrices on the host
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::X)], Width());
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::Y)], Height());
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::Z)], Depth());

		// transfer CPU -> GPU
		_dTransformMatrices[static_cast<size_t>(Direction::X)].resize(Width() * Width());
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].resize(Height() * Height());
		_dTransformMatrices[static_cast<size_t>(Direction::Z)].resize(Depth() * Depth());

		_dTransformMatrices[static_cast<size_t>(Direction::X)].set(_hTransformMatrices[static_cast<size_t>(Direction::X)].data(), Width() * Width());
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].set(_hTransformMatrices[static_cast<size_t>(Direction::Y)].data(), Height() * Height());
		_dTransformMatrices[static_cast<size_t>(Direction::Z)].set(_hTransformMatrices[static_cast<size_t>(Direction::Z)].data(), Depth() * Depth());
	}
}

void HartleyTransform::ForwardTransform(double* data) {
	PROFILE_FUNCTION();

	bool is1D = (Height() == 0 && Depth() == 0);
	bool is2D = (Height() > 0 && Depth() == 0);
	bool is3D = (Depth() > 0);

	// Проверяем, инициализирован ли MPI
	int mpiInitialized = 0;
#ifdef HAVE_MPI
	MPI_Initialized(&mpiInitialized);
#endif

	int rank = 0, size = 1;
	if (mpiInitialized) {
#ifdef HAVE_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
	}

	if (is1D || is2D) {
		// Для 1D и 2D нет смысла в MPI
		switch (_mode) {
		case Modes::CPU:
			if (is1D) {
				FDHT1D(data);
			} else {
				FDHT2D(data);
			}
			break;
		case Modes::GPU:
			if (is1D) {
				DHT1DCuda(data);
			} else {
				DHT2DCuda(data);
			}
			break;
		case Modes::RFFT:
			if (is1D) {
				RealFFT1D(data);
			} else {
				FDHT2D(data);
			}

			break;
		}
		if (mpiInitialized) {
#ifdef HAVE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		}
		return;
	}

	// 3D case: делим по Z между процессами, если MPI включен
	size_t depthPerProc = Depth() / size;
	size_t remainder = Depth() % size;
	size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
	depthPerProc += (rank < remainder) ? 1 : 0;

	double* localData = data + offset * Width() * Height();

	switch (_mode) {
	case Modes::CPU:
		FDHT3D(localData);
		break;
	case Modes::GPU:
		DHT3DCuda(localData);
		break;
	case Modes::RFFT:
		FDHT3D(localData);
		break;
	}

	// Сбор данных только если MPI активен
	if (mpiInitialized) {
#ifdef HAVE_MPI
		std::vector<int> sendcounts(size);
		std::vector<int> displs(size);
		int offs = 0;
		for (int i = 0; i < size; ++i) {
			sendcounts[i] = static_cast<int>((Depth() / size + (i < remainder ? 1 : 0)) * Width() * Height());
			displs[i] = offs;
			offs += sendcounts[i];
		}
		MPI_Allgatherv(localData, sendcounts[rank], MPI_DOUBLE,
			data, sendcounts.data(), displs.data(), MPI_DOUBLE,
			MPI_COMM_WORLD);
#endif
	}
}

void HartleyTransform::InverseTransform(double* data) {
	PROFILE_FUNCTION();

	bool is1D = (Height() == 0 && Depth() == 0);
	bool is2D = (Height() > 0 && Depth() == 0);
	bool is3D = (Depth() > 0);

	// Проверяем, инициализирован ли MPI
	int mpiInitialized = 0;
#ifdef HAVE_MPI
	MPI_Initialized(&mpiInitialized);
#endif

	int rank = 0, size = 1;
	if (mpiInitialized) {
#ifdef HAVE_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
	}

	// Сначала выполняем прямое преобразование
	ForwardTransform(data);

	// Общий размер данных
	size_t totalSize = Width();
	if (Height() > 0) {
		totalSize *= Height();
	}
	if (Depth() > 0) {
		totalSize *= Depth();
	}

	auto denominator = 1.0 / static_cast<double>(totalSize);

	if (is1D || is2D) {
		// Масштабируем полностью только на rank=0
		for (size_t i = 0; i < totalSize; ++i) {
			data[i] *= denominator;
		}
		if (mpiInitialized) {
#ifdef HAVE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		}
		return;
	}

	// 3D case: делим по Z между процессами
	size_t depthPerProc = Depth() / size;
	size_t remainder = Depth() % size;
	size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
	depthPerProc += (rank < remainder) ? 1 : 0;

	size_t localSize = depthPerProc * Width() * Height();
	double* localData = data + offset * Width() * Height();

	// Масштабируем локальный блок
	for (size_t i = 0; i < localSize; ++i) {
		localData[i] *= denominator;
	}

	// Синхронизация процессов
	if (mpiInitialized) {
#ifdef HAVE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}
}

void HartleyTransform::BitReverse(std::vector<size_t>& indices) {
	PROFILE_FUNCTION();

	if (indices.empty()) {
		return;
	}

	const size_t n = indices.size();
	const int kLog2n = static_cast<int>(std::log2(n));

	indices[0] = 0;
	for (size_t j = 1; j < n; ++j) {
		size_t reversed = 0;
		size_t temp = j;
		for (int i = 0; i < kLog2n; ++i) {
			if (temp & 1) {
				reversed |= 1 << (kLog2n - 1 - i);
			}
			temp >>= 1;
		}
		indices[j] = reversed;
	}
}