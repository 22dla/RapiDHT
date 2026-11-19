#!/usr/bin/env bash
# R-4 step 4A, diagnosis: why the y pass at 256^3 (pipelined kernel, work
# buffer) got 27 % slower with the twiddle folded into step 1, although its
# registers went down (96 -> 87) and its blocks per SM did not change (two).
# Same counters for the hybrid of 4B (A, from git stash) and for 4A (B, the
# working tree), on that one kernel: throughput, memory and shared-memory
# traffic with bank conflicts, warp stall reasons, occupancy, issue.
#
#   bash scripts/r4_diag_4a.sh
#
# Writes benchmarks/results/<date>_<host>_r4_s4a_diag_{a,b}.txt. Needs sudo
# for the counters (asked once per run unless cached).
set -euo pipefail
cd "$(dirname "$0")/.."

out="benchmarks/results/$(date +%F)_$(hostname)_r4_s4a_diag"
files="src/internal/factored_kernels.cuh tests/test_factored_kernels.cpp"

measure() {
    local tag=$1
    echo "== $tag: build"
    cmake --build build-bench -j > /dev/null
    echo "== $tag: ncu, y pass at 256^3"
    # At 256^3 only the y pass runs FactoredAxisKernelPipelined (z runs the
    # in-place kernel, x the fused one), so the first launch it matches is y.
    sudo -E env "PATH=$PATH" ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
        --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section WarpStateStats \
        --section SchedulerStats --section Occupancy --section LaunchStats \
        --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum \
        -k regex:FactoredAxisKernelPipelined -c 1 build-bench/benchmarks/profile_gpu3d 256 1 f32 factored \
        > "${out}_${tag}.txt"
}

git stash push $files
if ! measure a; then
    echo "A failed; restoring the working tree" >&2
fi
git stash pop
measure b
echo "== done: ${out}_{a,b}.txt"
