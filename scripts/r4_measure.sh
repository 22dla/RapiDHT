#!/usr/bin/env bash
# One measurement of the R-4 track, the way docs/R4_PLAN.md requires it.
#
#   R4_NO_CUFFT=1 bash scripts/r4_measure.sh <tag> ...   # sweeps: no cuFFT rows
#   R4_FULL=1 bash scripts/r4_measure.sh <tag>            # step 5.2: all sizes, 2D too
#   bash scripts/r4_measure.sh <tag> [VAR=value ...]
#
#   bash scripts/r4_measure.sh base                           # the code as it stands
#   bash scripts/r4_measure.sh wide RAPIDHT_R4_X=wide         # same build, a temporary switch
#
# Temporary RAPIDHT_R4_* switches live only until their step is decided (rule 4
# of docs/R4_PLAN.md); since step 5.1 the code has none.
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
#   <date>_<host>_r4_<tag>_trace.txt   time per kernel at 128^3, 256^3, 512^3,
#                                      with the SM clock and temperature
#                                      before and after each size ("# gpu" lines)
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
    echo "table: $([[ ${R4_FULL:-0} == 1 ]] && echo full || echo 128-512)"
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

# The trace goes first, on a cool card, and logs the clock around each size:
# taken right after the benchmark's 7 repetitions, the 512^3 trace moved by
# 10-25 % on kernels whose code had not changed (2026-11-15 s4a_a, 2026-11-19
# s4as), while the benchmark agreed with itself.
gpu_line() {
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v n="$1" -v when="$2" \
            '{ printf "n=%s # gpu %s sm_mhz=%s temp_c=%s power_w=%s throttle=%s\n", n, when, $1, $2, $3, $4 }' || true
}

# The SM clock is locked for the trace. The A-A check of 2026-11-19 (aa1,
# aa2: the same code twice) showed why: at 512^3 every kernel took one of two
# times, 4.34 or 3.84 ms on x, as the card went from 1665 MHz to its boost of
# 1935-1950, and the switch came after 5 calls in one run and 13 in the other
# -- 13 % on the total from the clock alone. 1665 MHz is the clock the card
# holds below boost, so locking there costs no heat and needs no warm-up. The
# benchmark is left unlocked so that its figures stay comparable with the
# earlier ones (it runs long enough to sit at boost). Needs sudo; without it the
# trace is taken unlocked and says so.
trace_clock_mhz=1665
locked=0
if sudo nvidia-smi -lgc "$trace_clock_mhz,$trace_clock_mhz" >/dev/null; then
    locked=1
    trap 'sudo nvidia-smi -rgc >/dev/null || true' EXIT
fi

echo "== 4/5 time per kernel (before the benchmark, 15 calls a size)"
: >"${stem}_trace.txt"
if [[ $locked == 1 ]]; then
    echo "# sm clock locked at $trace_clock_mhz MHz for the trace" | tee -a "${stem}_trace.txt"
else
    echo "# sm clock NOT locked (no sudo): per-kernel figures at 512^3 may move by 10-15 %" | tee -a "${stem}_trace.txt"
fi
for n in 128 256 512; do
    gpu_line "$n" before | tee -a "${stem}_trace.txt"
    env "$@" RAPIDHT_TRACE_KERNELS=1 "$profile" "$n" 15 f32 factored 2>&1 | sed "s/^/n=$n /" | tee -a "${stem}_trace.txt"
    gpu_line "$n" after | tee -a "${stem}_trace.txt"
done

if [[ $locked == 1 ]]; then
    sudo nvidia-smi -rgc >/dev/null || true
    trap - EXIT
fi

echo "== 5/5 benchmark: cuFFT and the factored path, f32, 7 repetitions (clock not locked)"
# R4_NO_CUFFT=1 leaves cuFFT out -- for sweeps that compare RapiDHT variants
# with each other, about 40 s a run. Keep it in for A-B runs and published
# ratios: cuFFT is the same program in A and B, so its time is a free A-A
# check of the conditions, and a ratio is only fair from the same session.
#
# R4_FULL=1 is the full table of step 5.2: 2D and 3D, 64^3-1024^3 and
# 512^2-2048^2, about 10 minutes. Without it only 128^3-512^3 are run, as in
# every A-B of the track (the sizes 64^3, 1024^3 and 2048^2 were added to
# bench_transform at step 5.2).
sizes='(128x128x128|256x256x256|512x512x512)$'
families='(cuFFT|cuFFT_R2C|FactoredResident|FactoredResidentSeparable)'
if [[ ${R4_NO_CUFFT:-0} == 1 ]]; then
    families='(FactoredResident|FactoredResidentSeparable)'
    echo "   (cuFFT left out: R4_NO_CUFFT=1)"
fi
filter="^$families/f32/3D/$sizes"
if [[ ${R4_FULL:-0} == 1 ]]; then
    filter="^$families/f32/(2D/(512x512|1024x1024|2048x2048)|3D/(64x64x64|128x128x128|256x256x256|512x512x512|1024x1024x1024))\$"
    echo "   (full table: R4_FULL=1)"
fi
env "$@" "$bench" \
    --benchmark_filter="$filter" \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true \
    --benchmark_out="$stem.json" --benchmark_out_format=json

echo
echo "done: $stem.json"
echo "compare: python3 tools/r4_compare.py <base>.json $stem.json --trace <base>_trace.txt ${stem}_trace.txt"
