#!/usr/bin/env python3
"""Find the M:1 win/lose boundary of the un-fuse rule vs intermediate size.

Intermediate row bytes at m=1: N * (K/32) * 4 = N*K/8. Sweep shapes across
that axis (dynamic delegate shapes, threads=4), interleaved A/B, min of 2,
tracking wall and process CPU.
"""
import subprocess, os, json, time

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = "bazel-bin/bench/subgraph/fully_connected"
CPUS = "6-11"

# (K, N) sorted by intermediate bytes N*K/8
CASES = [
    (256, 256),    # 8KB   (known lose)
    (512, 256),    # 16KB
    (256, 512),    # 16KB
    (1024, 128),   # 16KB  (contiguous exactly 512B)
    (512, 512),    # 32KB
    (1024, 256),   # 32KB
    (2048, 128),   # 32KB
    (1024, 512),   # 64KB
    (2048, 256),   # 64KB
    (4096, 128),   # 64KB
    (2048, 512),   # 128KB (known win)
    (4096, 256),   # 128KB
    (4096, 512),   # 256KB
    (8192, 256),   # 256KB
    (8192, 512),   # 512KB
]


def run(k, n, unfuse, min_time="0.3s"):
    env = dict(os.environ)
    env["YNN_UNFUSE"] = "1" if unfuse else "0"
    env.pop("YNN_XNNPACK_STATIC_SHAPES", None)
    out = f"{SP}/last2.json"
    r = subprocess.run(
        ["taskset", "-c", CPUS, BENCH,
         f"--benchmark_filter=QD8F32QB4WFullyConnected/M:1/K:{k}/N:{n}/",
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
        r = run(2048, 512, False, "0.2s")
        if r and r[0] < 40:  # healthy m1 K2048 N512 default ~= 15us
            return
        print(f"  (canary {r}, waiting)", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    results = {}
    print(f"{'K':>6} {'N':>6} {'int_KB':>7} {'t off':>8} {'t on':>8} "
          f"{'speedup':>8} {'cpu off':>8} {'cpu on':>8} {'cpu rat':>8}",
          flush=True)
    for k, n in CASES:
        canary()
        offs, ons = [], []
        for _ in range(2):
            a = run(k, n, False)
            b = run(k, n, True)
            if a: offs.append(a)
            if b: ons.append(b)
        if not offs or not ons:
            print(f"{k:>6} {n:>6} FAILED", flush=True)
            continue
        toff = min(o[0] for o in offs); coff = min(o[1] for o in offs)
        ton = min(o[0] for o in ons); con = min(o[1] for o in ons)
        ib = n * k // 8 // 1024
        results[f"{k}x{n}"] = {"int_kb": ib, "time_off": toff,
                               "time_on": ton, "cpu_off": coff, "cpu_on": con}
        print(f"{k:>6} {n:>6} {ib:>7} {toff:>7.2f}u {ton:>7.2f}u "
              f"{toff/ton:>7.2f}x {coff:>7.1f}u {con:>7.1f}u "
              f"{con/coff:>7.2f}x", flush=True)
    json.dump(results, open(f"{SP}/m1_boundary.json", "w"), indent=1)
    open(f"{SP}/DONE2", "w").write("done\n")
    print("M1 BOUNDARY DONE", flush=True)
