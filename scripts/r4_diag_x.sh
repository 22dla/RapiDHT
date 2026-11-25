#!/usr/bin/env bash
# R-4, toward a margin on P2+: what bounds the last pass at 512^3.
#
# After step 4.Z the x pass with Bracewell's recombination
# (FactoredLastAxisBracewellKernelInPlace: 8 lines, one tile a thread, two
# blocks to an SM, groups of four mirror lines) takes 3.62 ms -- the slowest of
# the three, 1.4 times the copy floor. Its lines run along x, so its loads are
# whole 2 KiB rows: the contiguity that cost z does not apply. This profiles
# the three passes of one 512^3 transform (z, y, x in that order, `-c 3`) side
# by side: duration, DRAM bytes and throughput, SM throughput and issue
# activity, occupancy and the warp stall reasons, so x can be compared against
# the two passes whose bounds are known (z: memory; y: overlapped).
#
#   bash scripts/r4_diag_x.sh
#
# Writes benchmarks/results/<date>_<host>_r4_x512_ncu.txt. Needs sudo for the
# counters; takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

out="benchmarks/results/$(date +%F)_$(hostname)_r4_x512_ncu.txt"
echo "== build"
cmake --build build-bench -j --target profile_gpu3d > /dev/null
echo "== ncu: z, y and x passes at 512^3"
sudo -E env "PATH=$PATH" ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis --section WarpStateStats --section Occupancy --section LaunchStats \
    --section SchedulerStats \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    -k 'regex:Factored' -c 3 \
    build-bench/benchmarks/profile_gpu3d 512 1 f32 factored > "$out"
echo "== done: $out"
grep -E '^  Factored|gpu__time_duration.sum|dram__bytes_(read|write).sum|DRAM Throughput|Compute \(SM\) Throughput|Achieved Occupancy|Registers Per Thread' "$out" \
    | sed -E 's/^  (Factored[A-Za-z]*)<[^(]*/  \1<...>/'
