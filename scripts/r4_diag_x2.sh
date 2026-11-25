#!/usr/bin/env bash
# R-4, toward a margin on P2+: where the x pass with Bracewell spends its issue
# slots (second look; the first is scripts/r4_diag_x.sh).
#
# 2026-11-21_*_r4_x512_ncu.txt: at 512^3 the Bracewell pass is neither
# memory- nor compute-bound -- DRAM 58 %, issue slots 61 %, 1.2 eligible warps a
# scheduler with 16 warps an SM -- yet it issues about 1.5 times the
# instructions of the y pass over the same data (6.0 M cycles x 61 % against
# 4.7 M x 50 %). This profiles the y pass and the x pass of one transform
# (`-c 3`, the first is z) for: instructions executed by kind (InstructionStats)
# and the average stall reason per issued instruction, so the extra work and
# what the warps wait on can be named before anything is changed.
#
#   bash scripts/r4_diag_x2.sh
#
# Writes benchmarks/results/<date>_<host>_r4_x512_ncu2.txt. Needs sudo for the
# counters; takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

out="benchmarks/results/$(date +%F)_$(hostname)_r4_x512_ncu2.txt"
echo "== build"
cmake --build build-bench -j --target profile_gpu3d > /dev/null
echo "== ncu: instruction mix and stall reasons, z / y / x at 512^3"
sudo -E env "PATH=$PATH" ncu --section InstructionStats \
    --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,'regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio' \
    -k 'regex:Factored' -c 3 \
    build-bench/benchmarks/profile_gpu3d 512 1 f32 factored > "$out"
echo "== done: $out"
