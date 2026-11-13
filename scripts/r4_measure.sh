#!/usr/bin/env bash
# One measurement of the R-4 track, the way docs/R4_PLAN.md requires it.
#
#   bash scripts/r4_measure.sh <tag> [VAR=value ...]
#
#   bash scripts/r4_measure.sh base                           # the code as it stands
#   bash scripts/r4_measure.sh v2 RAPIDHT_FACTORED_KERNEL=v2  # same build, another kernel
#
# Why a script: on 2026-11-13 a "v2.3" result was recorded from binaries built a
# week earlier, because the build step was skipped by hand. Here the build, the
# correctness gate and the record of what was measured cannot be skipped: the
# benchmark runs only after both build directories are up to date and the
# factored tests pass, and every result file gets a .meta.txt beside it with the
# commit, the uncommitted changes, the hashes of the device sources and the
# state of the GPU.
#
# Writes into benchmarks/results/:
#   <date>_<host>_r4_<tag>.json        medians of 7 repetitions (bench_transform)
#   <date>_<host>_r4_<tag>_trace.txt   time per kernel at 128^3, 256^3, 512^3
#   <date>_<host>_r4_<tag>.meta.txt    what exactly was measured
#
# Compare two runs with tools/r4_compare.py.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || "$1" == *=* ]]; then
    echo "usage: bash scripts/r4_measure.sh <tag> [VAR=value ...]" >&2
    exit 2
fi
tag=$1
shift
for assignment in "$@"; do
    if [[ "$assignment" != RAPIDHT_*=* ]]; then
        echo "only RAPIDHT_*=value settings are accepted, got: $assignment" >&2
        exit 2
    fi
done

results=benchmarks/results
stem="$results/$(date +%F)_$(hostname)_r4_${tag}"
bench=build-bench/benchmarks/bench_transform
profile=build-bench/benchmarks/profile_gpu3d
mkdir -p "$results"

if [[ -e "$stem.json" ]]; then
    echo "$stem.json already exists: pick another tag, results are never overwritten" >&2
    exit 2
fi

echo "== 1/5 build (never measure a stale binary)"
cmake --build --preset cuda -j
cmake --build build-bench -j

echo "== 2/5 correctness gate: the factored tests, host and device"
ctest --preset cuda -R 'Factored' --output-on-failure

echo "== 3/5 record what is being measured"
{
    echo "tag: $tag"
    echo "date: $(date --iso-8601=seconds)"
    echo "settings: ${*:-(none)}"
    echo "commit: $(git rev-parse HEAD)"
    echo "uncommitted:"
    git status --short | sed 's/^/  /'
    echo "sources:"
    sha256sum src/internal/factored_kernels.cuh src/kernels.cu src/backend_cuda.cpp src/internal/kernels.h \
        benchmarks/bench_transform.cpp benchmarks/cufft_baseline.cu | sed 's/^/  /'
    echo "binaries:"
    ls -l --time-style=long-iso "$bench" "$profile" | sed 's/^/  /'
    echo "gpu:"
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,memory.used --format=csv,noheader | sed 's/^/  /'
    echo "other processes on the gpu (they add noise; close what you can):"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | cut -c1-120 | sed 's/^/  /'
} >"$stem.meta.txt"
cat "$stem.meta.txt"

echo "== 4/5 benchmark: cuFFT and the factored path, f32 volumes, 7 repetitions"
env "$@" "$bench" \
    --benchmark_filter='^(cuFFT|cuFFT_R2C|FactoredResident|FactoredResidentSeparable)/f32/3D/' \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true \
    --benchmark_out="$stem.json" --benchmark_out_format=json

echo "== 5/5 time per kernel"
: >"${stem}_trace.txt"
for n in 128 256 512; do
    env "$@" RAPIDHT_TRACE_KERNELS=1 "$profile" "$n" 9 f32 factored 2>&1 | sed "s/^/n=$n /" | tee -a "${stem}_trace.txt"
done

echo
echo "done: $stem.json"
echo "compare: python3 tools/r4_compare.py <base>.json $stem.json --trace <base>_trace.txt ${stem}_trace.txt"
