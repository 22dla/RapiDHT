/*
 * Project: RapiDHT
 * File: src/internal/integer_offset.h
 * Brief: R-5 step 3.1 -- what an integer volume has subtracted on reading.
 *
 * The transform of an integer volume subtracts one integer c from every voxel
 * as its first pass reads it and adds c * N back to the zero coefficient at
 * the end: the result is the transform of the input itself, but the passes
 * work on values near zero, which takes the largest error of a large mean
 * away (docs/R5_PLAN.md, section 3). Which c is a policy:
 *
 *   None          c = 0
 *   Value(v)      c = v, an integer the caller knows (air, the matrix, half
 *                 the range)
 *   Mean          the rounded mean of the whole volume -- one more read of
 *                 it (0.70 ms at 512^3 uint16, 7.5 % of the transform)
 *   SampledMean   the rounded mean of every step-th line along x; every 64th
 *                 is the default (step 0.5: the cheapest policy within the
 *                 accuracy rule on every input; 18 us at 512^3 uint16)
 *
 * Mean and SampledMean are computed on the device, where the first pass reads
 * c, with no round trip through the host. Internal until the API decision of
 * step 6.1.
 */

#ifndef RAPIDHT_INTERNAL_INTEGER_OFFSET_H
#define RAPIDHT_INTERNAL_INTEGER_OFFSET_H

#include <cstddef>

namespace RapiDHT {
namespace internal {

struct IntegerOffset {
    enum class Kind { None,
        Value,
        Mean,
        SampledMean };

    Kind kind = Kind::None;
    double value = 0.0; ///< Kind::Value: the integer subtracted
    size_t step = 1; ///< Kind::SampledMean: every step-th line along x

    static IntegerOffset NoOffset() { return {}; }
    static IntegerOffset Of(double v) { return { Kind::Value, v, 1 }; }
    static IntegerOffset MeanOfAll() { return { Kind::Mean, 0.0, 1 }; }
    static IntegerOffset MeanOfEvery(size_t lineStep) { return { Kind::SampledMean, 0.0, lineStep }; }
    /// The default policy (docs/R5_PLAN.md, step 0.5): every 64th line.
    static IntegerOffset Default() { return MeanOfEvery(64); }
};

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_INTEGER_OFFSET_H
