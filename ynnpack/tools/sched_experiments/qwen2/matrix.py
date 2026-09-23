#!/usr/bin/env python3
"""Qwen3-0.6B matrix on composites blockwise dot_bench: default vs un-fuse rule.

Static shapes native. One case per process, pinned, interleaved, min of 2.
"""
import subprocess, os, json, time

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = f"{ROOT}/bazel-bin/ynnpack/composites/dot_bench"
CPUS = "6-11"

NK = [(1024, 1024), (2048, 1024), (1024, 2048), (3072, 1024), (1024, 3072)]
MS = [1, 2, 4, 512]
TYPES = ["blockwise_int4_bs32", "blockwise_int8_bs32", "blockwise_int4_bs256"]

CASES = [(t, m, n, k) for t in TYPES for m in MS for n, k in NK]
CASES += [("blockwise_int4_bs32", 1, 151936, 1024),
          ("blockwise_int4_bs32", 4, 151936, 1024)]


def run(typ, m, n, k, unfuse, min_time="0.3s"):
    env = dict(os.environ, YNN_UNFUSE="1" if unfuse else "0")
    out = f"{SP}/last.json"
    r = subprocess.run(
        ["taskset", "-c", CPUS, BENCH, "--benchmarks=blockwise",
         f"-m={m}", f"-n={n}", f"-k={k}", "--thread_count=4",
         f"--benchmark_filter={typ}/", f"--benchmark_min_time={min_time}",
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
        r = run("blockwise_int4_bs32", 1, 1024, 1024, False, "0.1s")
        if r and r[0] < 30:  # healthy default ~= 12us
            return
        print(f"  (canary {r}, waiting)", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    results = {}
    print(f"{'case':<40} {'t off':>9} {'t on':>9} {'speedup':>8} {'cpu rat':>8}",
          flush=True)
    for i, (typ, m, n, k) in enumerate(CASES):
        if i % 10 == 0:
            canary()
        offs, ons = [], []
        for _ in range(2):
            a = run(typ, m, n, k, False)
            b2 = run(typ, m, n, k, True)
            if a: offs.append(a)
            if b2: ons.append(b2)
        label = f"{typ.replace('blockwise_', '')}/m{m}/n{n}/k{k}"
        if not offs or not ons:
            print(f"{label:<40} FAILED", flush=True)
            continue
        toff = min(o[0] for o in offs); coff = min(o[1] for o in offs)
        ton = min(o[0] for o in ons); con = min(o[1] for o in ons)
        results[label] = {"t_off": toff, "t_on": ton, "c_off": coff, "c_on": con}
        print(f"{label:<40} {toff:>8.1f}u {ton:>8.1f}u {toff/ton:>7.2f}x "
              f"{con/coff:>7.2f}x", flush=True)
    json.dump(results, open(f"{SP}/matrix.json", "w"), indent=1)
    open(f"{SP}/DONE", "w").write("done\n")
    print("MATRIX DONE", flush=True)
