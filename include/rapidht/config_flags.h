/*
 * Project: RapiDHT
 * File: include/rapidht/config_flags.h
 * Brief: Which backends this build carries, and whether a device is present.
 */

#ifndef RAPIDHT_CONFIG_FLAGS_H
#define RAPIDHT_CONFIG_FLAGS_H

#include <rapidht/config.h>

namespace RapiDHT {

/// True when the library was built with the CUDA backend (Modes::GPU usable).
#ifdef RAPIDHT_WITH_CUDA
inline constexpr bool kCudaEnabled = true;
#else
inline constexpr bool kCudaEnabled = false;
#endif

/// True when the library was built with the MPI-distributed 3D backend.
#ifdef RAPIDHT_WITH_MPI
inline constexpr bool kMpiEnabled = true;
#else
inline constexpr bool kMpiEnabled = false;
#endif

/**
 * @brief Whether a usable CUDA device is actually present, right now.
 *
 * kCudaEnabled reports only that the backend was compiled in, which is a
 * different question: a machine can carry the toolkit and no card at all, which
 * is the normal state of a build server. Anything that will touch the device --
 * Modes::GPU, DeviceVolume -- needs this one, and it is the check to make
 * before offering the GPU backend to a user.
 *
 * Never throws; a driver too old for the runtime simply reports false.
 */
bool IsGpuAvailable() noexcept;

} // namespace RapiDHT

#endif // RAPIDHT_CONFIG_FLAGS_H
