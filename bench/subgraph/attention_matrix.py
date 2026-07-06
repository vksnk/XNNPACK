#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Builds and runs the attention benchmark comparison matrix:
#
#   1. xnnpack native            (//bench/subgraph:attention, FP32Attention)
#   2. xnnpack + ynnpack         (same, --define=xnnpack_use_ynnpack=true)
#   3. ynnpack composite vanilla (//ynnpack/composites:attention_benchmark)
#   4. ynnpack composite flash   (same, FlashAttention256)
#
# Each configuration runs in a fresh process, pinned to the fastest cores
# (detected from lscpu), interleaved across rounds to average out frequency
# drift. Reports median time, GFLOP/s and max RSS as markdown tables.
#
# Run from the repository root:
#
#   python3 bench/subgraph/attention_matrix.py [--rounds=3] [--threads=1,8]
#       [--seqs=1024,4096] [--flash-w=256] [--skip-build] [--out=results.json]
#
# Notes:
# - The composite benchmark has thread counts compiled in (1,2,4,8,16,32);
#   --threads values outside that set are skipped for the composites.
# - MODULE.bazel references slinky by absolute path; the machine needs that
#   checkout (same branch/patches) or an edited path.
# - Builds are pinned to clang (--cc to override): gcc 16 compiles the ynn
#   dot microkernels ~1.7x slower than clang 22 (IPC 2.4 vs 3.7), which is
#   invisible in the graph/IR and once masqueraded as a machine-level
#   bimodality. The compiler is recorded in the report header.

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

HEAD_DIM = 64
NUM_HEADS = 32


def sh(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def detect_cores():
    """Returns CPU ids sorted by descending max frequency (stable by id)."""
    r = sh(["lscpu", "-e=CPU,MAXMHZ"])
    cores = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                cores.append((int(parts[0]), float(parts[1])))
            except ValueError:
                cores.append((int(parts[0]), 0.0))
    cores.sort(key=lambda c: (-c[1], c[0]))
    return [c[0] for c in cores]


def machine_info(cc):
    info = {"hostname": sh(["hostname"]).stdout.strip(),
            "date": datetime.now().isoformat(timespec="seconds"),
            "cc": sh([cc, "--version"]).stdout.splitlines()[0]}
    for line in sh(["lscpu"]).stdout.splitlines():
        for key in ("Model name", "CPU(s)", "Thread(s) per core"):
            if line.startswith(key):
                info[key] = line.split(":", 1)[1].strip()
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["governor"] = gov.read_text().strip()
    except OSError:
        pass
    load = Path("/proc/loadavg").read_text().split()[0]
    info["loadavg"] = load
    r = sh(["git", "rev-parse", "--short", "HEAD"])
    info["git"] = r.stdout.strip()
    return info


def build(stage_dir, cc):
    toolchain = f"--repo_env=CC={cc}"
    targets = [
        ("xnn_native", ["bazel", "build", "-c", "opt", toolchain,
                        "//bench/subgraph:attention"],
         "bazel-bin/bench/subgraph/attention"),
        # dynamic_mode=off: the staged copy must not depend on bazel-bin .so
        # runfiles.
        ("composite", ["bazel", "build", "-c", "opt", toolchain,
                       "--dynamic_mode=off",
                       "//ynnpack/composites:attention_benchmark"],
         "bazel-bin/ynnpack/composites/attention_benchmark"),
        # Last: this build overwrites bazel-bin outputs of the first.
        ("xnn_ynn", ["bazel", "build", "-c", "opt", toolchain,
                     "--define=xnnpack_use_ynnpack=true",
                     "//bench/subgraph:attention"],
         "bazel-bin/bench/subgraph/attention"),
    ]
    binaries = {}
    for name, cmd, output in targets:
        print(f"building {name}...", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"build failed: {' '.join(cmd)}")
        dst = stage_dir / name
        # Staged copies inherit bazel's read-only mode; remove before
        # overwriting.
        dst.unlink(missing_ok=True)
        shutil.copy2(output, dst)
        binaries[name] = dst
    return binaries


# Runs the benchmark under a nested interpreter so ru_maxrss of
# RUSAGE_CHILDREN reflects only this benchmark process: in this process it
# would be a cumulative high-water mark across all previous runs.
_WRAPPER = """
import resource, subprocess, sys
r = subprocess.run(sys.argv[1:], capture_output=True, text=True)
sys.stdout.write(r.stdout + r.stderr)
print("MAXRSS_KB", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
"""


def run_one(binary, args, pin_cpus):
    """Runs one benchmark process pinned to pin_cpus; returns parsed result."""
    cmd = ["taskset", "-c", ",".join(map(str, pin_cpus)),
           sys.executable, "-c", _WRAPPER, str(binary)] + args
    # Sample the pinned core's frequency while the benchmark runs. Frequency
    # governors respond differently to different thread pool idle patterns
    # (e.g. spinning vs. sleeping workers), which can silently skew results;
    # reporting the observed frequency makes that visible.
    freq_path = Path(f"/sys/devices/system/cpu/cpu{pin_cpus[0]}/cpufreq/"
                     "scaling_cur_freq")
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append(int(freq_path.read_text()))
            except OSError:
                return
            stop.wait(0.1)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    r = sh(cmd)
    stop.set()
    t.join(timeout=1)
    out = {"rss_mb": 0.0, "raw": None}
    if samples:
        out["observed_ghz"] = statistics.median(samples) / 1e6
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("MAXRSS_KB"):
            out["rss_mb"] = int(line.split()[1]) / 1024.0
            continue
        if "/process_time/real_time" not in line:
            continue
        out["raw"] = line.strip()
        m = re.search(r"real_time\s+(\d+)\s+(ns|us|ms)", line)
        if m:
            scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3}[m.group(2)]
            out["seconds"] = int(m.group(1)) * scale
        m = re.search(r"cpufreq=([0-9.]+)G", line)
        if m:
            out["cpufreq_ghz"] = float(m.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--threads", default="1,8")
    p.add_argument("--seqs", default="1024,4096")
    p.add_argument("--flash-w", type=int, default=256)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--cc", default="clang")
    p.add_argument("--out", default="attention_matrix_results.json")
    args = p.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    seqs = [int(s) for s in args.seqs.split(",")]

    info = machine_info(args.cc)
    print("# Attention benchmark matrix")
    for k, v in info.items():
        print(f"#   {k}: {v}")
    if float(info.get("loadavg", 0)) > 1.0:
        print("# WARNING: load average > 1, results may be depressed by "
              "background load / boost contention")

    cores = detect_cores()
    stage_dir = Path(tempfile.gettempdir()) / "attention_matrix_stage"
    stage_dir.mkdir(exist_ok=True)
    if args.skip_build:
        binaries = {n: stage_dir / n for n in ("xnn_native", "composite",
                                               "xnn_ynn")}
        missing = [n for n, b in binaries.items() if not b.exists()]
        if missing:
            sys.exit(f"--skip-build but staged binaries missing: {missing}")
    else:
        binaries = build(stage_dir, args.cc)

    def variants(seq, th):
        xnn_filter = f"^FP32Attention/T:{seq}/H:{HEAD_DIM}/N:{NUM_HEADS}/S:{seq}/"
        comp = f"seq:{seq}/head:{HEAD_DIM}/heads:{NUM_HEADS}"
        return [
            ("xnn native vanilla", "xnn_native",
             [f"--benchmark_filter={xnn_filter}", f"--num_threads={th}"]),
            ("xnn+ynnpack vanilla", "xnn_ynn",
             [f"--benchmark_filter={xnn_filter}", f"--num_threads={th}"]),
            ("ynn composite vanilla", "composite",
             [f"--benchmark_filter=^Attention/{comp}/threads:{th}/"]),
            (f"ynn composite flash w{args.flash_w}", "composite",
             [f"--benchmark_filter=^FlashAttention{args.flash_w}/{comp}/"
              f"threads:{th}/"]),
        ]

    results = {}  # (variant, seq, th) -> list of runs
    for rnd in range(args.rounds):
        for seq in seqs:
            for th in threads:
                pin = cores[:max(th, 1)]
                for label, binary, bargs in variants(seq, th):
                    r = run_one(binaries[binary], bargs, pin)
                    if r["raw"] is None:
                        print(f"# WARNING: no result for {label} seq={seq} "
                              f"threads={th} (round {rnd})")
                        continue
                    results.setdefault((label, seq, th), []).append(r)
                    ghz = r.get("observed_ghz")
                    freq = f", {ghz:.1f} GHz" if ghz else ""
                    print(f"# round {rnd}: {label} seq={seq} {th}T: "
                          f"{r['seconds']*1e3:.1f} ms, RSS {r['rss_mb']:.0f} MB"
                          f"{freq}", flush=True)

    # Markdown tables, one per sequence length.
    report = []
    for seq in seqs:
        flops = 4 * NUM_HEADS * seq * seq * HEAD_DIM  # QK^T and P@V
        report.append(f"\n## seq={seq}, h={HEAD_DIM}, {NUM_HEADS} heads "
                      f"({flops / 1e9:.1f} GFLOP/invoke)\n")
        header = "| variant | " + " | ".join(
            f"{th}T" for th in threads) + " | " + " | ".join(
            f"RSS {th}T" for th in threads) + " |"
        report.append(header)
        report.append("|---" * (1 + 2 * len(threads)) + "|")
        for label, _, _ in variants(seq, threads[0]):
            row = [label]
            for th in threads:
                runs = results.get((label, seq, th))
                if not runs:
                    row.append("n/a")
                    continue
                secs = statistics.median(r["seconds"] for r in runs)
                row.append(f"{secs*1e3:.0f} ms ({flops/secs/1e9:.0f} GF/s)")
            for th in threads:
                runs = results.get((label, seq, th))
                row.append(f"{max(r['rss_mb'] for r in runs):.0f} MB"
                           if runs else "n/a")
            report.append("| " + " | ".join(row) + " |")
    print("\n".join(report))

    with open(args.out, "w") as f:
        json.dump({"machine": info,
                   "results": [{"variant": k[0], "seq": k[1], "threads": k[2],
                                "runs": v} for k, v in results.items()]},
                  f, indent=2)
    print(f"\n# raw results written to {args.out}")
    print(f"# staged binaries in {stage_dir} (reusable with --skip-build)")


if __name__ == "__main__":
    main()
