#!/usr/bin/env python3
"""Baseline FP32LayerNorm masks 1..7: ynn default scheduler vs stock xnn."""
import subprocess, os, json

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
YNN = f"{ROOT}/bazel-bin/bench/subgraph/layer_norm"
XNN = os.path.join(os.path.dirname(SP), "layer_norm_xnn")
CPUS = "6-11"


def run(binary, mask, threads=4, min_time="0.5s"):
    env = dict(os.environ, YNN_UNFUSE="0")
    out = f"{SP}/last.json"
    subprocess.run(
        ["taskset", "-c", CPUS, binary,
         f"--benchmark_filter=FP32LayerNorm/M:128/N:256/K:512/NormMask:{mask}/",
         f"--num_threads={threads}", f"--benchmark_min_time={min_time}",
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


if __name__ == "__main__":
    print(f"{'mask':>4} {'ynn t':>10} {'ynn cpu':>10} {'xnn t':>10} "
          f"{'xnn cpu':>10} {'ynn/xnn':>8}", flush=True)
    res = {}
    for mask in range(1, 8):
        y = min((run(YNN, mask) for _ in range(2)), key=lambda r: r[0] if r else 1e18)
        x = min((run(XNN, mask) for _ in range(2)), key=lambda r: r[0] if r else 1e18)
        res[mask] = {"ynn": y, "xnn": x}
        print(f"{mask:>4} {y[0]:>9.1f}u {y[1]:>9.1f}u {x[0]:>9.1f}u "
              f"{x[1]:>9.1f}u {y[0]/x[0]:>7.2f}x", flush=True)
    json.dump(res, open(f"{SP}/baseline.json", "w"), indent=1)
    print("BASELINE DONE", flush=True)
