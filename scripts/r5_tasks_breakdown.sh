#!/usr/bin/env bash
# R-5 step 5.2: where the time of each end-to-end task goes, kernel by kernel,
# on both sides -- this library's passes and cuFFT's own kernels, which only a
# profiler sees. One nsys process per task row (Task<N>/... of bench_transform,
# 512^3 uint16, input on the device), a fixed number of iterations, and the
# per-kernel summary of each:
#
#   bash scripts/r5_tasks_breakdown.sh <tag> [size]      # size default 512x512x512
#
# Writes benchmarks/results/<date>_<host>_r5_<tag>_breakdown.txt. The times are
# nsys's (clock not locked, profiler attached): use them for shares, and take
# the totals of a task from an r5_measure.sh run (R5_TASKS=1). Each row also runs
# its correctness check once and its set-up (kernel spectrum, plans); those
# kernels show up with 1-2 instances, the timed ones with about 30 + 2.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "usage: bash scripts/r5_tasks_breakdown.sh <tag> [size]" >&2
    exit 2
fi
tag=$1
size=${2:-512x512x512}
bench=build-bench/benchmarks/bench_transform
out="benchmarks/results/$(date +%F)_$(hostname)_r5_${tag}_breakdown.txt"
if [[ -e "$out" ]]; then
    echo "$out already exists: pick another tag" >&2
    exit 2
fi
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

cmake --build build-bench -j >/dev/null

rows=(
    Task1/IntFactored/u16/device
    Task1/cuFFTLto/u16/device
    Task2/Hartley/u16/s2/device
    Task2/HartleySeparable/u16/s2/device
    Task2/cuFFTLto/u16/s2/device
    Task2/Spatial/u16/s2/device
    Task3/Hartley/u16/device
    Task3/HartleySeparable/u16/device
    Task3/cuFFTLto/u16/device
)
{
    echo "# r5_tasks_breakdown.sh $tag, $(date --iso-8601=seconds), commit $(git rev-parse --short HEAD), size $size"
    echo "# $(nsys --version)"
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader | sed 's/^/# gpu: /'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/# other process: /'
} >"$out"
for row in "${rows[@]}"; do
    name="$row/3D/$size"
    echo "== $name" | tee -a "$out"
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o "$scratch/run" \
        "$bench" --benchmark_filter="^${name}\$" --benchmark_min_time=30x --benchmark_repetitions=1 \
        2>&1 | grep -E "^Task" | tee -a "$out"
    nsys stats --quiet --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format table "$scratch/run.nsys-rep" \
        2>/dev/null | cut -c1-200 | tee -a "$out"
done
echo "wrote $out"
