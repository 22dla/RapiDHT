/*
 * Project: RapiDHT
 * File: src/internal/integer_offset.h
 * Brief: R-5 -- the offset policy under its internal name.
 *
 * The policy became public at step 6.1 (<rapidht/integer_offset.h>); the
 * internal entry points, benchmarks and tests of stages 2-5 keep spelling it
 * internal::IntegerOffset through this alias.
 */

#ifndef RAPIDHT_INTERNAL_INTEGER_OFFSET_H
#define RAPIDHT_INTERNAL_INTEGER_OFFSET_H

#include <rapidht/integer_offset.h>

namespace RapiDHT {
namespace internal {

using IntegerOffset = RapiDHT::IntegerOffset;

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_INTEGER_OFFSET_H
