#!/usr/bin/env python3
"""Autotuner campaign on FP32LayerNorm masks: record -> sweep -> random ->
hillclimb per mask, stock scheduler (YNN_UNFUSE=0), threads=4, ONE lane
(bandwidth-bound: keep the other CCD idle)."""
import subprocess, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
TUNE = f"{ROOT}/ynnpack/tools/autotune.py"
BENCH = f"{ROOT}/bazel-bin/bench/subgraph/layer_norm"

MASKS = [int(x) for x in sys.argv[1].split(",")]
CPUS = sys.argv[2]

env = dict(os.environ, YNN_UNFUSE="0")


def bench_cmd(mask, min_time="0.3s"):
    return [BENCH,
            f"--benchmark_filter=FP32LayerNorm/M:128/N:256/K:512/NormMask:{mask}/",
            "--num_threads=4", f"--benchmark_min_time={min_time}"]


def tune(args):
    subprocess.run([sys.executable, TUNE] + args, env=env, cwd=ROOT)


for mask in MASKS:
    space = f"{SP}/space_m{mask}.txt"
    csv = f"{SP}/results_m{mask}.csv"
    print(f"=== mask {mask} record ===", flush=True)
    if not os.path.exists(space):
        tune(["record", "-o", space, "--"] +
             ["taskset", "-c", CPUS] + bench_cmd(mask, "0.05s"))
    print(f"=== mask {mask} sweep ===", flush=True)
    tune(["sweep", "--space", space, "-o", csv, "--cpus", CPUS, "--"] +
         bench_cmd(mask))
    print(f"=== mask {mask} random ===", flush=True)
    tune(["random", "--space", space, "-o", csv, "--cpus", CPUS,
          "--samples", "100", "--seed", str(mask), "--"] + bench_cmd(mask))
    print(f"=== mask {mask} hillclimb ===", flush=True)
    tune(["hillclimb", "--space", space, "-o", csv, "--cpus", CPUS,
          "--steps", "60", "--seed", str(100 + mask), "--"] + bench_cmd(mask))
    print(f"=== mask {mask} done ===", flush=True)

open(f"{SP}/DONE_{'_'.join(map(str, MASKS))}", "w").write("done\n")
print("CAMPAIGN LANE DONE", flush=True)
