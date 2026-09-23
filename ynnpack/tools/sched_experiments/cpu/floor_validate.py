#!/usr/bin/env python3
"""Validate the 16KB/thread work floor: regressions gone, wins intact."""
import subprocess, os, json, time

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = "bazel-bin/bench/subgraph/fully_connected"
CPUS = "6-11"

# (label, M, K, N, static)
CASES = [
    ("REGR dyn m1 K256 N256",   1, 256, 256, False),
    ("REGR sta m1 K256 N256",   1, 256, 256, True),
    ("floor dyn m1 K512 N512",  1, 512, 512, False),   # 32KB: must not fire
    ("edge dyn m1 K1024 N512",  1, 1024, 512, False),  # 64KB: fires, breakeven
    ("win dyn m1 K2048 N512",   1, 2048, 512, False),
    ("win dyn m16 K2048 N512",  16, 2048, 512, False),
    ("win dyn m64 K2048 N512",  64, 2048, 512, False),
    ("win dyn m256 K2048 N512", 256, 2048, 512, False),
    ("win sta m32 K2048 N512",  32, 2048, 512, True),
    ("win sta m64 K2048 N512",  64, 2048, 512, True),
]


def run(m, k, n, static, unfuse, min_time="0.3s"):
    env = dict(os.environ)
    env["YNN_UNFUSE"] = "1" if unfuse else "0"
    if static:
        env["YNN_XNNPACK_STATIC_SHAPES"] = "1"
    else:
        env.pop("YNN_XNNPACK_STATIC_SHAPES", None)
    out = f"{SP}/last3.json"
    r = subprocess.run(
        ["taskset", "-c", CPUS, BENCH,
         f"--benchmark_filter=QD8F32QB4WFullyConnected/M:{m}/K:{k}/N:{n}/",
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


def canary():
    while True:
        r = run(1, 2048, 512, True, False, "0.2s") or run(1, 2048, 512, False, False, "0.2s")
        if r and r[0] < 40:
            return
        print(f"  (canary {r}, waiting)", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    print(f"{'case':<26} {'t off':>9} {'t on':>9} {'speedup':>8} "
          f"{'cpu rat':>8}", flush=True)
    for label, m, k, n, static in CASES:
        canary()
        offs, ons = [], []
        for _ in range(2):
            a = run(m, k, n, static, False)
            b = run(m, k, n, static, True)
            if a: offs.append(a)
            if b: ons.append(b)
        if not offs or not ons:
            print(f"{label:<26} FAILED", flush=True)
            continue
        toff = min(o[0] for o in offs); coff = min(o[1] for o in offs)
        ton = min(o[0] for o in ons); con = min(o[1] for o in ons)
        print(f"{label:<26} {toff:>8.2f}u {ton:>8.2f}u {toff/ton:>7.2f}x "
              f"{con/coff:>7.2f}x", flush=True)
    open(f"{SP}/DONE3", "w").write("done\n")
    print("FLOOR VALIDATE DONE", flush=True)
