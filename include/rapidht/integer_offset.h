/*
 * Project: RapiDHT
 * File: include/rapidht/integer_offset.h
 * Brief: What a transform of integer voxels subtracts before the passes.
 */

#ifndef RAPIDHT_INTEGER_OFFSET_H
#define RAPIDHT_INTEGER_OFFSET_H

#include <cstddef>

namespace RapiDHT {

/**
 * @brief The integer c a transform of uint8 or uint16 voxels works around.
 *
 * The forward transform of an integer volume subtracts c from every voxel as
 * it reads it and adds c * N back to the zero-frequency coefficient at the
 * end, so the result is the transform of the input itself. The passes then
 * work on values near zero, which removes the largest rounding error of a
 * volume whose mean is far from zero -- a tomogram around 30000 in uint16 is
 * the typical case. The inverse into integers takes c * N out of H(0) before
 * its passes and adds c back to every voxel as it writes it.
 *
 * In float the choice of c moves results by about one float step; a mean
 * near the data's own is what keeps that small (docs/benchmarks.md, R-5).
 *
 * Policies:
 *  - NoOffset(): c = 0.
 *  - Of(v): c = v, an integer the caller knows (air, the matrix, half the
 *    range); no extra read.
 *  - MeanOfAll(): the rounded mean of the whole volume -- one more read of
 *    the input for the forward transform; for the inverse, rint(H(0) / N),
 *    the rounded mean of the volume being restored, which costs nothing.
 *  - MeanOfEvery(step): the rounded mean of every step-th line along the
 *    fastest axis; for the inverse the same as MeanOfAll().
 *  - Default(): MeanOfEvery(64), the cheapest policy within the accuracy
 *    rule on every input measured (about 0.2 % of a 512^3 transform).
 *
 * The means are computed on the device in the device modes, with no round
 * trip through the host.
 */
struct IntegerOffset {
    /// Which policy; read the factory functions rather than setting it.
    enum class Kind { None,
        Value,
        Mean,
        SampledMean };

    Kind kind = Kind::None;
    double value = 0.0; ///< Kind::Value: the integer subtracted.
    size_t step = 1; ///< Kind::SampledMean: every step-th line along x (0 counts as 1).

    /// c = 0.
    static IntegerOffset NoOffset() { return {}; }
    /// c = v, a value the caller knows.
    static IntegerOffset Of(double v) { return { Kind::Value, v, 1 }; }
    /// The rounded mean of the whole volume.
    static IntegerOffset MeanOfAll() { return { Kind::Mean, 0.0, 1 }; }
    /// The rounded mean of every `lineStep`-th line along the fastest axis.
    static IntegerOffset MeanOfEvery(size_t lineStep) { return { Kind::SampledMean, 0.0, lineStep }; }
    /// The default: MeanOfEvery(64).
    static IntegerOffset Default() { return MeanOfEvery(64); }
};

} // namespace RapiDHT

#endif // RAPIDHT_INTEGER_OFFSET_H
