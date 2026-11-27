#!/usr/bin/env bash
# One measurement of the R-6 track (faster GPU kernels: convolution, passes of
# length 1024, integer edge passes), the way docs/R6_PLAN.md requires it. A copy
# of scripts/r5_measure.sh -- same build, gate, record and trace -- with the R-6
# rows:
#
#   bash scripts/r6_measure.sh <tag> [RAPIDHT_*=value ...]
#   R6_LARGE=1 bash scripts/r6_measure.sh <tag>    # plus 1024x1024x512 (bench and trace)
#
# Every run times the R-4 rows (rule 8 of the R-5 plan), the convolutions
# (Conv*, direction A), and the integer transform against its strongest cuFFT
# opponent (IntFactoredEvery64 / cuFFTIntLto, direction C), 128^3-512^3.
#
# Writes into benchmarks/results/:
#   <date>_<host>_r6_<tag>.json            medians of 7 repetitions (bench_transform)
#   <date>_<host>_r6_<tag>_trace.txt       time per kernel: f32 (n=<size>), u16 (n=<size>/u16),
#                                          with R6_LARGE=1 also n=1024x1024x512
#   <date>_<host>_r6_<tag>.meta.txt        what exactly was measured
#
# Compare two runs with tools/r4_compare.py.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || "$1" == *=* ]]; then
    echo "usage: bash scripts/r6_measure.sh <tag> [VAR=value ...]" >&2
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
stem="$results/$(date +%F)_$(hostname)_r6_${tag}"
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
ctest --preset cuda -R 'Factored|Integer|Convolution' --output-on-failure

echo "== 3/5 record what is being measured"
{
    echo "tag: $tag"
    echo "date: $(date --iso-8601=seconds)"
    echo "settings: ${*:-(none)}"
    echo "table: 128-512$([[ ${R6_LARGE:-0} == 1 ]] && echo ' + 1024x1024x512')"
    echo "cuda: $(nvcc --version | sed -n 's/.*release \([^,]*\), V\(.*\)/\1 (\2)/p')"
    echo "commit: $(git rev-parse HEAD)"
    echo "uncommitted:"
    git status --short | sed 's/^/  /'
    echo "sources:"
    sha256sum src/internal/factored_kernels.cuh src/kernels.cu src/backend_cuda.cpp src/internal/kernels.h \
        benchmarks/bench_transform.cpp benchmarks/cufft_baseline.h benchmarks/cufft_baseline.cu \
        benchmarks/cufft_common.cuh benchmarks/cufft_int_callback.cu benchmarks/cufft_int_lto_callback.cu \
        benchmarks/profile_gpu3d.cpp | sed 's/^/  /'
    for optional in benchmarks/convolution_gpu.h benchmarks/convolution_gpu.cu src/convolution.cpp \
        include/rapidht/convolution.h; do
        [[ -e "$optional" ]] && sha256sum "$optional" | sed 's/^/  /'
    done
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
# Float labelled n=<size> as in R-4, u16 as n=<size>/u16 as in R-5.
trace_sizes=(128 256 512)
[[ ${R6_LARGE:-0} == 1 ]] && trace_sizes+=(1024x1024x512)
for type in f32 u16; do
    for n in "${trace_sizes[@]}"; do
        label="n=$n"
        [[ $type == f32 ]] || label="n=$n/$type"
        gpu_line "$label" before | tee -a "${stem}_trace.txt"
        env "$@" RAPIDHT_TRACE_KERNELS=1 "$profile" "$n" 15 "$type" factored 2>&1 | sed "s|^|$label |" | tee -a "${stem}_trace.txt"
        gpu_line "$label" after | tee -a "${stem}_trace.txt"
    done
done

if [[ $locked == 1 ]]; then
    sudo nvidia-smi -rgc >/dev/null || true
    trap - EXIT
fi

echo "== 5/5 benchmark: R-4 rows, convolutions, integer transform, 7 repetitions (clock not locked)"
#   Conv{Hartley,HartleyEven,Cufft,CufftEven}  convolution, any kernel / even kernel (R-4 step 5.3)
#   IntFactoredEvery64 / cuFFTIntLto           the integer transform and its strongest opponent (R-5)
sizes='(128x128x128|256x256x256|512x512x512)'
r4_rows="(cuFFT|cuFFT_R2C|FactoredResident|FactoredResidentSeparable)/f32/3D/$sizes"
conv_sizes=$sizes
int_sizes=$sizes
if [[ ${R6_LARGE:-0} == 1 ]]; then
    r4_rows="(cuFFT|cuFFT_R2C|FactoredResident|FactoredResidentSeparable)/f32/3D/(128x128x128|256x256x256|512x512x512|1024x1024x512)"
    conv_sizes='(128x128x128|256x256x256|512x512x512|1024x1024x512)'
    int_sizes=$conv_sizes
    echo "   (large sizes: R6_LARGE=1)"
fi
conv_rows="(ConvHartley|ConvHartleyEven|ConvCufft|ConvCufftEven)/f32/3D/$conv_sizes"
int_rows="(IntFactoredEvery64|cuFFTIntLto)/(u8|u16)/3D/$int_sizes"
filter="^($r4_rows|$conv_rows|$int_rows)\$"
env "$@" "$bench" \
    --benchmark_filter="$filter" \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true \
    --benchmark_out="$stem.json" --benchmark_out_format=json

echo
echo "done: $stem.json"
echo "compare: python3 tools/r4_compare.py <base>.json $stem.json --trace <base>_trace.txt ${stem}_trace.txt"
