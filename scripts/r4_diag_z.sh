#!/usr/bin/env bash
# R-4, toward P2+: why the z pass at 512^3 is 0.9 ms slower than y.
#
# Since step 4.Y both passes run the same kernel instantiation
# (FactoredAxisKernelInPlace, 16 lines, two tiles a thread, one block an SM),
# so the arithmetic is identical and only the stride of a line differs:
# 2 KiB along y, 1 MiB along z. The z pass is the first in-place launch of
# the transform and y the second, so `-c 2` profiles exactly that pair.
#
# Counters, per kernel: duration; DRAM bytes read and written against the
# 512 MiB that one pass must move each way (more means overfetch, the same at a
# lower rate means the DRAM is used inefficiently); L2 hit rate and sectors per
# request; DRAM and L2 throughput; warp stall reasons.
#
#   bash scripts/r4_diag_z.sh
#
# Writes benchmarks/results/<date>_<host>_r4_z512_ncu.txt. Needs sudo for the
# counters; takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

out="benchmarks/results/$(date +%F)_$(hostname)_r4_z512_ncu.txt"
echo "== build"
cmake --build build-bench -j --target profile_gpu3d > /dev/null
echo "== ncu: z and y passes at 512^3"
sudo -E env "PATH=$PATH" ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section Occupancy \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum \
    -k 'regex:FactoredAxisKernelInPlace' -c 2 \
    build-bench/benchmarks/profile_gpu3d 512 1 f32 factored > "$out"
echo "== done: $out"
grep -E 'FactoredAxisKernelInPlace|gpu__time_duration.sum|dram__bytes_(read|write).sum|lts__t_sector_hit_rate.pct' "$out" \
    | sed -E 's/FactoredAxisKernelInPlace<[^(]*/FactoredAxisKernelInPlace<...>/'
