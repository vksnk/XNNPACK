#!/bin/bash
# run_case.sh <name> <m> <n> <k> <cpus>
set -e
NAME=$1; M=$2; N=$3; K=$4; CPUS=$5
SP=$(cd "$(dirname "$0")" && pwd)
cd ~/Projects/XNNPACK
BENCH="bazel-bin/ynnpack/composites/dot_bench --thread_count=4 --shape=$M,$N,$K --benchmark_filter=blockwise_int4_bs32"
if [ ! -f $SP/$NAME.space ]; then
  YNN_SCHED_TUNE=record:$SP/$NAME.space $BENCH --benchmark_min_time=1x >/dev/null 2>&1
fi
# The un-fuse rule replay: v17.fuse=0 (+ out3.d1.scale=/4 when the m loop exists).
awk '{ if ($1=="v17.fuse") $4=0; if ($1=="out3.d1.scale") $4=4; print }' $SP/$NAME.space > $SP/$NAME.rule
python3 ynnpack/tools/autotune.py random --space $SP/$NAME.space --samples 80 --mutate-prob 0.3 \
  --cpus $CPUS -o $SP/$NAME.csv -- $BENCH --benchmark_min_time=0.3s --benchmark_repetitions=3 \
  > $SP/$NAME.log 2>&1
D=$(taskset -c $CPUS $BENCH --benchmark_min_time=0.5s 2>/dev/null | grep -m1 real_time | awk '{print $2}')
R=$(YNN_SCHED_TUNE=replay:$SP/$NAME.rule taskset -c $CPUS $BENCH --benchmark_min_time=0.5s 2>/dev/null | grep -m1 real_time | awk '{print $2}')
echo "$NAME m=$M n=$N k=$K default=${D}ns rule=${R}ns"
