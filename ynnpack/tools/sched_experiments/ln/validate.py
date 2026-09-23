#!/usr/bin/env python3
"""Validate LayerNorm campaign winners (interleaved A/B) and ablate mask 3.

Uses the per-mask results CSV for winner choices, replays via YNN_SCHED_TUNE.
"""
import csv as csvmod
import json, os, subprocess

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = f"{ROOT}/bazel-bin/bench/subgraph/layer_norm"
CPUS = "6-11"


def load_space(mask):
    space = []
    for line in open(f"{SP}/space_m{mask}.txt"):
        p = line.split()
        if p[0] == "pipeline":
            space.append(("pipeline", p[1]))
        else:
            space.append((p[0], p[1], int(p[2]), int(p[3])))
    return space


def winner(mask):
    best = None
    for row in csvmod.DictReader(open(f"{SP}/results_m{mask}.csv")):
        if not row.get("time_ns"):
            continue
        t = float(row["time_ns"])
        if best is None or t < best[0]:
            best = (t, json.loads(row["choices_json"]))
    return best[1]


def write_replay(mask, choices, path):
    lines = []
    pipe = None
    for e in load_space(mask):
        if e[0] == "pipeline":
            pipe = e[1]
            lines.append(f"pipeline {pipe}")
        else:
            k, kind, n, dflt = e
            c = choices.get(f"{pipe}/{k}", dflt)
            lines.append(f"{k} {kind} {n} {c}")
    open(path, "w").write("\n".join(lines) + "\n")


def run(mask, choices, min_time="0.4s"):
    replay = f"{SP}/replay_val.txt"
    write_replay(mask, choices, replay)
    env = dict(os.environ, YNN_UNFUSE="0",
               YNN_SCHED_TUNE=f"replay:{replay}")
    out = f"{SP}/last_val.json"
    subprocess.run(
        ["taskset", "-c", CPUS, BENCH,
         f"--benchmark_filter=FP32LayerNorm/M:128/N:256/K:512/NormMask:{mask}/",
         "--num_threads=4", f"--benchmark_min_time={min_time}",
         "--benchmark_format=json", f"--benchmark_out={out}",
         "--benchmark_out_format=json"],
        env=env, capture_output=True, text=True, cwd=ROOT)
    try:
        data = json.load(open(out))
    except Exception:
        return None
    for b in data.get("benchmarks", []):
        if b.get("run_type", "iteration") == "iteration":
            unit = b.get("time_unit", "ns")
            scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[unit]
            return (b["real_time"] * scale, b["cpu_time"] * scale)
    return None


def ab(mask, choices, rounds=3):
    d, w = [], []
    for _ in range(rounds):
        a = run(mask, {})
        b = run(mask, choices)
        if a: d.append(a)
        if b: w.append(b)
    td = min(x[0] for x in d); cd = min(x[1] for x in d)
    tw = min(x[0] for x in w); cw = min(x[1] for x in w)
    return td, tw, cd, cw


if __name__ == "__main__":
    print("== winner validation (interleaved, min of 3 x 0.4s) ==", flush=True)
    winners = {}
    for mask in [3, 7, 4, 5, 6]:
        ch = winner(mask)
        winners[mask] = ch
        td, tw, cd, cw = ab(mask, ch)
        nd = len(ch)
        print(f"mask {mask}: default {td:8.1f}u -> {tw:8.1f}u  "
              f"{td/tw:5.2f}x  cpu {cw/cd:4.2f}x  ({nd} nondefault)", flush=True)

    print("\n== mask 3 ablation (leave-one-out from winner) ==", flush=True)
    ch3 = winners[3]
    td, tw, cd, cw = ab(3, ch3, rounds=2)
    print(f"full winner: {tw:.1f}u ({td/tw:.2f}x)", flush=True)
    for k in sorted(ch3):
        sub = {kk: vv for kk, vv in ch3.items() if kk != k}
        _, t, _, c = ab(3, sub, rounds=2)
        print(f"  -{k.split('/',1)[1]:<18} {t:8.1f}u (delta {t - tw:+7.1f}u)",
              flush=True)

    print("\n== mask 3 singletons ==", flush=True)
    for k in sorted(ch3):
        single = {k: ch3[k]}
        _, t, _, c = ab(3, single, rounds=2)
        print(f"  only {k.split('/',1)[1]:<15} {t:8.1f}u ({td/t:.2f}x)",
              flush=True)

    print("\n== mask 7 v7.fuse option sweep ==", flush=True)
    pipe7 = [e[1] for e in load_space(7) if e[0] == "pipeline"][0]
    for opt in range(4):
        _, t, _, c = ab(7, {f"{pipe7}/v7.fuse": opt}, rounds=2)
        print(f"  v7.fuse={opt}: {t:8.1f}u", flush=True)
    open(f"{SP}/DONE_VAL", "w").write("done\n")
    print("VALIDATE DONE", flush=True)
