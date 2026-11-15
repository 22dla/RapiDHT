#!/usr/bin/env bash
# R-4 step 4B, diagnosis: why the in-place steps are slower at 128^3 and on
# the fused x pass, and faster on y and z at 512^3. Same counters for the code
# of step 3.3 (A, from git stash) and of 4B (B, the working tree):
#
#   - registers and local memory per kernel (cuobjdump, no sudo);
#   - per kernel at 128^3 and 512^3 (ncu, needs sudo for the counters):
#     launch configuration, achieved occupancy, warp stall reasons (barrier
#     among them), issue slots, throughput, and local-memory traffic, which
#     shows register spills.
#
#   bash scripts/r4_diag_4b.sh
#
# Writes benchmarks/results/<date>_<host>_r4_s4b_diag_{a,b}_{res,ncu_128,ncu_512}.txt.
# Takes a few minutes; asks for the sudo password once per ncu run unless it is
# cached.
set -euo pipefail
cd "$(dirname "$0")/.."

out="benchmarks/results/$(date +%F)_$(hostname)_r4_s4b_diag"
files="src/kernels.cu src/internal/factored_kernels.cuh tests/test_factored_kernels.cpp"
kernels='regex:Factored(AxisKernelPipelined|LastAxisBracewell)'

measure() {
    local tag=$1
    echo "== $tag: build"
    cmake --build build-bench -j > /dev/null
    echo "== $tag: registers"
    cuobjdump -res-usage build-bench/benchmarks/profile_gpu3d 2> /dev/null \
        | grep -A1 -E 'Factored(AxisKernelPipelined|LastAxisBracewell)' > "${out}_${tag}_res.txt" || true
    for n in 128 512; do
        echo "== $tag: ncu at ${n}^3 (z, y, x)"
        sudo -E env "PATH=$PATH" ncu --section LaunchStats --section Occupancy --section WarpStateStats \
            --section SchedulerStats --section SpeedOfLight \
            --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum \
            -k "$kernels" -c 3 build-bench/benchmarks/profile_gpu3d "$n" 1 f32 factored > "${out}_${tag}_ncu_${n}.txt"
    done
}

git stash push $files
if ! measure a; then
    echo "A failed; restoring the working tree" >&2
fi
git stash pop
measure b
echo "== done: ${out}_*"
