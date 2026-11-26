#!/usr/bin/env bash
# One measurement of the R-5 track (integer volumes), the way docs/R5_PLAN.md
# requires it. A copy of scripts/r4_measure.sh -- same build, gate, record and
# trace -- with the R-5 families added:
#
#   bash scripts/r5_measure.sh <tag> [RAPIDHT_*=value ...]
#   R5_ACC=1 bash scripts/r5_measure.sh <tag>      # plus accuracy_integer
#   R5_TASKS=1 bash scripts/r5_measure.sh <tag>    # end-to-end tasks (stage 5)
#   R5_LARGE=1 bash scripts/r5_measure.sh <tag>    # R-5 families also at 1024x1024x512, 1024^3
#   R5_BUS=1 bash scripts/r5_measure.sh <tag>      # plus host to device, 512^3 and 1024^3 (step 0.6)
#
# Every run also times the R-4 rows (FactoredResident, FactoredResidentSeparable,
# f32, 128^3-512^3) and cuFFT, so that rule 8 of the plan -- the float path must
# not get slower -- is checked in the same session. Until the R-5 families exist
# (steps 0.2, 0.3, 2.3) their part of the filter matches nothing; that is not an
# error.
#
# Writes into benchmarks/results/:
#   <date>_<host>_r5_<tag>.json            medians of 7 repetitions (bench_transform)
#   <date>_<host>_r5_<tag>_trace.txt       time per kernel at 128^3-512^3: f32 (n=<size>,
#                                          as in R-4), then u16 and u8 (n=<size>/<type>)
#   <date>_<host>_r5_<tag>_accuracy.txt    with R5_ACC=1: accuracy_integer's report
#   <date>_<host>_r5_<tag>_accuracy_table.txt  ... and its verdict (tools/r5_accuracy_table.py)
#   <date>_<host>_r5_<tag>.meta.txt        what exactly was measured, with the hashes
#                                          of the cuFFT baselines and of the data files
#
# Compare two runs with tools/r4_compare.py; it knows the R-5 families.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || "$1" == *=* ]]; then
    echo "usage: bash scripts/r5_measure.sh <tag> [VAR=value ...]" >&2
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
stem="$results/$(date +%F)_$(hostname)_r5_${tag}"
bench=build-bench/benchmarks/bench_transform
profile=build-bench/benchmarks/profile_gpu3d
accuracy=build-bench/benchmarks/accuracy_integer
# Real volumes read by this run (rule 10 of docs/R5_PLAN.md: path and SHA-256 of
# each go into .meta.txt; accuracy figures of two runs compare only when the
# hashes agree). The Korneev columns, cut by tools/r5_tiff_to_raw.py (step 0.4;
# the table of files and hashes is in docs/R5_PLAN.md). Hashed only with
# R5_ACC=1, the only time they are read.
korneev=${R5_KORNEEV:-$HOME/data/korneev}
r5_data=()
if [[ ${R5_ACC:-0} == 1 ]]; then
    r5_data=("$korneev/C6197A_center512.raw" "$korneev/C6203A_center512.raw"
             "$korneev/C6197A_full1024x512.raw" "$korneev/C6203A_full1024x512.raw")
    for file in "${r5_data[@]}"; do
        [[ -r "$file" ]] || { echo "R5_ACC=1: $file is missing (R5_KORNEEV= sets the folder)" >&2; exit 2; }
    done
fi
mkdir -p "$results"

if [[ -e "$stem.json" ]]; then
    echo "$stem.json already exists: pick another tag, results are never overwritten" >&2
    exit 2
fi

echo "== 1/5 build (never measure a stale binary)"
cmake --build --preset cuda -j
cmake --build build-bench -j

echo "== 2/5 correctness gate: the factored tests, host and device"
ctest --preset cuda -R 'Factored|Integer' --output-on-failure

echo "== 3/5 record what is being measured"
{
    echo "tag: $tag"
    echo "date: $(date --iso-8601=seconds)"
    echo "settings: ${*:-(none)}"
    echo "table: $([[ ${R5_TASKS:-0} == 1 ]] && echo tasks || echo 128-512)$([[ ${R5_LARGE:-0} == 1 ]] && echo ' + large')$([[ ${R5_BUS:-0} == 1 ]] && echo ' + bus')$([[ ${R5_ACC:-0} == 1 ]] && echo ' + accuracy')"
    echo "cuda: $(nvcc --version | sed -n 's/.*release \([^,]*\), V\(.*\)/\1 (\2)/p')"
    echo "commit: $(git rev-parse HEAD)"
    echo "uncommitted:"
    git status --short | sed 's/^/  /'
    echo "sources:"
    sha256sum src/internal/factored_kernels.cuh src/kernels.cu src/backend_cuda.cpp src/internal/kernels.h \
        benchmarks/bench_transform.cpp benchmarks/cufft_baseline.h benchmarks/cufft_baseline.cu \
        benchmarks/cufft_common.cuh benchmarks/cufft_int_callback.cu benchmarks/cufft_int_lto_callback.cu \
        benchmarks/accuracy_integer.cpp | sed 's/^/  /'
    # The end-to-end tasks of stage 5, once they exist.
    for optional in benchmarks/tasks_gpu.h benchmarks/tasks_gpu.cu; do
        [[ -e "$optional" ]] && sha256sum "$optional" | sed 's/^/  /'
    done
    echo "data:"
    if [[ ${#r5_data[@]} -eq 0 ]]; then
        echo "  (none: generated inputs only)"
    else
        for file in "${r5_data[@]}"; do
            sha256sum "$file" | sed 's/^/  /'
        done
    fi
    echo "binaries:"
    ls -l --time-style=long-iso "$bench" "$profile" "$accuracy" | sed 's/^/  /'
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
# Float first, labelled n=<size> as in R-4; then the integer path (step 2.4),
# labelled n=<size>/u16 and n=<size>/u8, so r4_compare.py keeps them apart.
for type in f32 u16 u8; do
    for n in 128 256 512; do
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

echo "== 5/5 benchmark: R-4 rows and the R-5 families, 7 repetitions (clock not locked)"
# Rule 8: the R-4 rows and cuFFT float run in every measurement. The R-5
# families (uint8 / uint16 input, float output):
#   cuFFTInt     conversion kernel into a float buffer + R2C + Hartley pass   (step 0.2a)
#   cuFFTIntCb   legacy load callback, static cuFFT                           (step 0.2b)
#   cuFFTIntLto  LTO load callback                                            (step 0.2c)
#   IntConvertFactored  conversion kernel + FactoredResident                  (step 0.3)
#   IntFactored  the first pass reads the integers itself                     (step 2.3)
#   IntFactored<Policy>  the same with the library's offset policy: None, Every512,
#                Every64 (the default), Mean -- IntFactored subtracts half the range (step 3.2)
#   IntOffset    the mean offset on the device: exact, every 64th / 512th line (step 1.2)
sizes='(128x128x128|256x256x256|512x512x512)'
r4_rows="(cuFFT|cuFFT_R2C|FactoredResident|FactoredResidentSeparable)/f32/3D/$sizes"
r5_sizes=$sizes
# R5_LARGE=1: where does each integer path stop fitting? The R-4 rows stay at
# 128^3-512^3. The CPU reference of the check takes about a minute at
# 1024x1024x512.
if [[ ${R5_LARGE:-0} == 1 ]]; then
    r5_sizes='(128x128x128|256x256x256|512x512x512|1024x1024x512|1024x1024x1024)'
    echo "   (large sizes: R5_LARGE=1)"
fi
r5_rows="(cuFFTInt|cuFFTIntCb|cuFFTIntLto|IntConvertFactored|IntFactored|IntFactored(None|Every512|Every64|Mean))/(u8|u16)/3D/$r5_sizes"
# 2D (step 2.6): the first present pass is y. The float rows of the same images
# alongside, for the gap to cuFFT that R-4 left in 2D.
sizes_2d='(512x512|1024x1024|2048x2048)'
r5_rows_2d="((cuFFTInt|cuFFTIntCb|cuFFTIntLto|IntConvertFactored|IntFactored)/(u8|u16)|(cuFFT|FactoredResident)/f32)/2D/$sizes_2d"
# The price of the mean offset (step 1.2): a few seconds, so in every run.
offset_rows="IntOffset/(u8|u16)/(mean|every64|every512)/3D/(512x512x512|1024x1024x1024)"
filter="^($r4_rows|$r5_rows|$r5_rows_2d|$offset_rows)\$"
# R5_BUS=1 (step 0.6): one way over the bus for uint8, uint16 and float, from
# ordinary and pinned host memory -- about a minute, so not in every run.
if [[ ${R5_BUS:-0} == 1 ]]; then
    filter="^($r4_rows|$r5_rows|$r5_rows_2d|$offset_rows|H2D/(u8|u16|f32)/(pageable|pinned)/3D/(512x512x512|1024x1024x1024))\$"
    echo "   (host to device: R5_BUS=1)"
fi
# R5_TASKS=1 is stage 5: the end-to-end tasks (Task<N>/<implementation>/...),
# with the R-4 rows kept for rule 8.
if [[ ${R5_TASKS:-0} == 1 ]]; then
    filter="^($r4_rows\$|Task)"
    echo "   (end-to-end tasks: R5_TASKS=1)"
fi
env "$@" "$bench" \
    --benchmark_filter="$filter" \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true \
    --benchmark_out="$stem.json" --benchmark_out_format=json

# Accuracy goes last: it does not depend on the clock, and the CPU reference in
# double at 512^3 takes minutes that would otherwise warm the card before the
# timing.
# The inputs (step 0.4): white noise; the synthetic cylinder, u16 and u8; the
# centre of each column (inside the field of view) as a 128^3 box and whole;
# a 128^3 box of each full slice on the diagonal, where the edge of the field
# of view crosses it (52 % zeros outside the circle -- the hard case), and the
# full slices whole (1024x1024x512, about two minutes each). Real volumes are uint16 only.
if [[ ${R5_ACC:-0} == 1 ]]; then
    echo "== accuracy (R5_ACC=1)"
    {
        echo "# accuracy_integer, $(date --iso-8601=seconds), commit $(git rev-parse --short HEAD)$( [[ -n $(git status --short --untracked-files=no) ]] && echo ' + uncommitted changes (see .meta.txt)'); settings: ${*:-(none)}"
        env "$@" "$accuracy" 128 256 512
        for type in u16 u8; do
            env "$@" "$accuracy" --synthetic cylinder $type 128 512
        done
        for column in C6197A C6203A; do
            env "$@" "$accuracy" --raw "$korneev/${column}_center512.raw" 512x512x512 u16 --crop 192 192 192 128 128 128
            env "$@" "$accuracy" --raw "$korneev/${column}_center512.raw" 512x512x512 u16
            env "$@" "$accuracy" --raw "$korneev/${column}_full1024x512.raw" 1024x1024x512 u16 --crop 86 86 192 128 128 128
            env "$@" "$accuracy" --raw "$korneev/${column}_full1024x512.raw" 1024x1024x512 u16
        done
    } 2>&1 | tee "${stem}_accuracy.txt"
    # The default offset policy and criteria C3 / C4 by the rules of the plan.
    python3 tools/r5_accuracy_table.py "${stem}_accuracy.txt" | tee "${stem}_accuracy_table.txt"
fi

echo
echo "done: $stem.json"
echo "compare: python3 tools/r4_compare.py <base>.json $stem.json --trace <base>_trace.txt ${stem}_trace.txt"
