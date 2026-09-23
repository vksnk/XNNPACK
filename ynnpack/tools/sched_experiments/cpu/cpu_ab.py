#!/usr/bin/env python3
"""A/B the un-fuse rule tracking BOTH real_time and process cpu_time.

One case per process, pinned to one CCD, canary-guarded, interleaved A/B.
"""
import subprocess, os, json, time

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = "bazel-bin/bench/subgraph/fully_connected"
CPUS = "6-11"

# (label, M, static, threads)
CASES = [
    ("static_m1",   1,   True, 4),
    ("static_m32",  32,  True, 4),
    ("static_m64",  64,  True, 4),
    ("static_m256", 256, True, 4),
    ("dyn_m1",      1,   False, 4),
    ("dyn_m16",     16,  False, 4),
    ("dyn_m64",     64,  False, 4),
    ("dyn_m256",    256, False, 4),
]


def run(m, static, threads, unfuse, min_time="0.4s"):
    env = dict(os.environ)
    env["YNN_UNFUSE"] = "1" if unfuse else "0"
    if static:
        env["YNN_XNNPACK_STATIC_SHAPES"] = "1"
    else:
        env.pop("YNN_XNNPACK_STATIC_SHAPES", None)
    out = f"{SP}/last.json"
    r = subprocess.run(
        ["taskset", "-c", CPUS, BENCH,
         f"--benchmark_filter=QD8F32QB4WFullyConnected/M:{m}/K:2048/N:512/",
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


def canary():
    while True:
        r = run(16, True, 4, False, "0.2s")
        if r and r[0] < 120:
            return
        print(f"  (canary {r}, waiting)", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    results = {}
    print(f"{'case':<12} {'time off':>9} {'time on':>9} {'speedup':>8} "
          f"{'cpu off':>9} {'cpu on':>9} {'cpu ratio':>9}", flush=True)
    for label, m, static, threads in CASES:
        canary()
        # interleave: off, on, off, on; keep min of each arm
        offs, ons = [], []
        for _ in range(2):
            a = run(m, static, threads, False)
            b = run(m, static, threads, True)
            if a: offs.append(a)
            if b: ons.append(b)
        if not offs or not ons:
            print(f"{label:<12} FAILED", flush=True)
            continue
        toff = min(o[0] for o in offs); coff = min(o[1] for o in offs)
        ton = min(o[0] for o in ons); con = min(o[1] for o in ons)
        results[label] = {"time_off": toff, "time_on": ton,
                          "cpu_off": coff, "cpu_on": con}
        print(f"{label:<12} {toff:>8.1f}u {ton:>8.1f}u {toff/ton:>7.2f}x "
              f"{coff:>8.1f}u {con:>8.1f}u {con/coff:>8.2f}x", flush=True)
    json.dump(results, open(f"{SP}/cpu_ab.json", "w"), indent=1)
    open(f"{SP}/DONE", "w").write("done\n")
    print("CPU A/B DONE", flush=True)
