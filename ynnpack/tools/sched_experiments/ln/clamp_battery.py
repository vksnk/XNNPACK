#!/usr/bin/env python3
"""Broad A/B battery for the step clamp (YNN_STEP_CLAMP=1).

Arms per case: off vs on. Interleaved, min of 2, one lane, canary-guarded.
Covers: LayerNorm masks, softmax masks, blockwise QB4W (static+dyn, with and
without the un-fuse rule to test interplay).
"""
import subprocess, os, json, time

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
CPUS = "6-11"
B = f"{ROOT}/bazel-bin/bench/subgraph"

# (label, binary, filter, extra_env)
CASES = [
    ("LN m3 dyn",    "layer_norm", "FP32LayerNorm/M:128/N:256/K:512/NormMask:3/", {}),
    ("LN m7 dyn",    "layer_norm", "FP32LayerNorm/M:128/N:256/K:512/NormMask:7/", {}),
    ("LN m1 dyn",    "layer_norm", "FP32LayerNorm/M:128/N:256/K:512/NormMask:1/", {}),
    ("LN m4 dyn",    "layer_norm", "FP32LayerNorm/M:128/N:256/K:512/NormMask:4/", {}),
    ("SM m1 dyn",    "softmax",    "FP32SoftmaxDecomp/M:128/N:256/K:512/NormMask:1/", {}),
    ("SM m3 dyn",    "softmax",    "FP32SoftmaxDecomp/M:128/N:256/K:512/NormMask:3/", {}),
    ("SM m4 dyn",    "softmax",    "FP32SoftmaxDecomp/M:128/N:256/K:512/NormMask:4/", {}),
    ("QB4 sta m32 U0", "fully_connected", "QD8F32QB4WFullyConnected/M:32/K:2048/N:512/",
     {"YNN_XNNPACK_STATIC_SHAPES": "1", "YNN_UNFUSE": "0"}),
    ("QB4 sta m32 U1", "fully_connected", "QD8F32QB4WFullyConnected/M:32/K:2048/N:512/",
     {"YNN_XNNPACK_STATIC_SHAPES": "1", "YNN_UNFUSE": "1"}),
    ("QB4 sta m64 U1", "fully_connected", "QD8F32QB4WFullyConnected/M:64/K:2048/N:512/",
     {"YNN_XNNPACK_STATIC_SHAPES": "1", "YNN_UNFUSE": "1"}),
    ("QB4 dyn m1 U1",  "fully_connected", "QD8F32QB4WFullyConnected/M:1/K:2048/N:512/",
     {"YNN_UNFUSE": "1"}),
    ("QB4 dyn m64 U1", "fully_connected", "QD8F32QB4WFullyConnected/M:64/K:2048/N:512/",
     {"YNN_UNFUSE": "1"}),
    ("QB4 dyn m256 U1", "fully_connected", "QD8F32QB4WFullyConnected/M:256/K:2048/N:512/",
     {"YNN_UNFUSE": "1"}),
    ("QB4 bs256 m1",   "fully_connected", "QD8F32QB8WFullyConnected/M:1/K:2048/N:512/",
     {"YNN_UNFUSE": "1"}),
]


def run(binary, filt, extra, clamp, min_time="0.4s"):
    env = dict(os.environ)
    env.pop("YNN_XNNPACK_STATIC_SHAPES", None)
    env.setdefault("YNN_UNFUSE", "0")
    env.update(extra)
    if clamp:
        env["YNN_STEP_CLAMP"] = "1"
    else:
        env.pop("YNN_STEP_CLAMP", None)
    out = f"{SP}/last_clamp.json"
    subprocess.run(
        ["taskset", "-c", CPUS, f"{B}/{binary}",
         f"--benchmark_filter={filt}", "--num_threads=4",
         f"--benchmark_min_time={min_time}", "--benchmark_format=json",
         f"--benchmark_out={out}", "--benchmark_out_format=json"],
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
    print(f"{'case':<18} {'t off':>9} {'t on':>9} {'speedup':>8} {'cpu rat':>8}",
          flush=True)
    for label, binary, filt, extra in CASES:
        offs, ons = [], []
        for _ in range(2):
            a = run(binary, filt, extra, False)
            b2 = run(binary, filt, extra, True)
            if a: offs.append(a)
            if b2: ons.append(b2)
        if not offs or not ons:
            print(f"{label:<18} FAILED", flush=True)
            continue
        toff = min(o[0] for o in offs); coff = min(o[1] for o in offs)
        ton = min(o[0] for o in ons); con = min(o[1] for o in ons)
        print(f"{label:<18} {toff:>8.1f}u {ton:>8.1f}u {toff/ton:>7.2f}x "
              f"{con/coff:>7.2f}x", flush=True)
    open(f"{SP}/DONE_CLAMP", "w").write("done\n")
    print("CLAMP BATTERY DONE", flush=True)
