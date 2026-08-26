#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Autotuner for ynnpack schedules.

Searches over the scheduler's decision space (see schedule_tuner.h) by
benchmarking replayed decision files, and reports the best schedules found
and how much headroom they have over the default heuristics.

Typical use, with a google-benchmark binary and a filter selecting ONE case:

  # 1. Record the search space of a benchmark case.
  autotune.py record -o space.txt -- ./reduce_bench --benchmark_filter=...

  # 2. Random search (also collects cost-model training data in the CSV).
  autotune.py random --space space.txt --samples 200 -o results.csv -- \\
      ./reduce_bench --benchmark_filter=...

  # 3. Coordinate sweep: change one decision at a time from the default.
  autotune.py sweep --space space.txt -o results.csv -- ./reduce_bench ...

  # 4. Report the best schedules from a results CSV.
  autotune.py report results.csv

  # Stress: run a (test) binary under N random schedules, report failures.
  autotune.py stress --seeds 20 -- ./dot_test

Every sample runs in a fresh process. Use --cpus to pin with taskset. The
results CSV accumulates across runs (samples already measured, identified by
their choice signature, are not re-run) and records the pipeline IR hash, so
distinct decision vectors producing identical schedules are benchmarked once.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile


class Decision:

  def __init__(self, key, kind, num_options, choice):
    self.key = key
    self.kind = kind
    self.num_options = num_options
    self.choice = choice


class Space:
  """The decision space recorded by YNN_SCHED_TUNE=record."""

  def __init__(self, path):
    self.pipelines = {}  # pipeline key (hex string) -> [Decision]
    with open(path) as f:
      current = None
      for line in f:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
          continue
        if parts[0] == "pipeline":
          current = self.pipelines.setdefault(parts[1], [])
        elif current is not None and len(parts) == 4:
          current.append(Decision(parts[0], parts[1], int(parts[2]),
                                  int(parts[3])))

  def decisions(self):
    for pipeline, decisions in self.pipelines.items():
      for d in decisions:
        yield pipeline, d

  def size(self):
    n = 1
    for _, d in self.decisions():
      n *= d.num_options
    return n

  def default_choices(self):
    return {(p, d.key): d.choice for p, d in self.decisions()}

  def write_choices(self, choices, path):
    with open(path, "w") as f:
      for pipeline, decisions in self.pipelines.items():
        f.write(f"pipeline {pipeline}\n")
        for d in decisions:
          choice = choices.get((pipeline, d.key), d.choice)
          f.write(f"{d.key} {d.kind} {d.num_options} {choice}\n")


def signature(choices):
  """Stable id of a choice assignment, used to dedupe samples."""
  return hashlib.sha256(
      json.dumps(sorted((p, k, c) for (p, k), c in choices.items()),
                 separators=(",", ":")).encode()).hexdigest()[:16]


def run_benchmark(cmd, choices, space, cpus=None, check=False):
  """Runs one sample; returns (times_by_benchmark, ir_hash) or (None, err)."""
  with tempfile.TemporaryDirectory() as tmp:
    replay = os.path.join(tmp, "choices.txt")
    out_json = os.path.join(tmp, "bench.json")
    ir = os.path.join(tmp, "ir.txt")
    space.write_choices(choices, replay)
    env = dict(os.environ)
    env["YNN_SCHED_TUNE"] = f"replay:{replay}"
    env["YNN_SCHED_IR"] = ir
    full_cmd = list(cmd) + [
        f"--benchmark_out={out_json}", "--benchmark_out_format=json"
    ]
    if cpus:
      full_cmd = ["taskset", "-c", cpus] + full_cmd
    result = subprocess.run(full_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
      return None, f"exit {result.returncode}: {result.stderr[-500:]}"
    ir_hash = ""
    if os.path.exists(ir):
      with open(ir, "rb") as f:
        ir_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    with open(out_json) as f:
      data = json.load(f)
    times = {}
    to_ns = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
    for b in data.get("benchmarks", []):
      scale = to_ns.get(b.get("time_unit", "ns"), 1.0)
      if b.get("run_type") == "aggregate":
        # Prefer the median aggregate when repetitions are used.
        if b.get("aggregate_name") == "median":
          times[b["run_name"]] = b["real_time"] * scale
      else:
        times.setdefault(b["name"], b["real_time"] * scale)
    return (times, ir_hash), None


RESULT_FIELDS = ["signature", "ir_hash", "time_ns", "n_nondefault",
                 "times_json", "choices_json"]


def load_results(path):
  results = {}
  if os.path.exists(path):
    with open(path) as f:
      for row in csv.DictReader(f):
        results[row["signature"]] = row
  return results


def append_result(path, row, write_header):
  with open(path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
    if write_header:
      writer.writeheader()
    writer.writerow(row)


def measure(cmd, choices, space, args, results, out_path, label):
  """Measures one choice assignment, with caching by signature and IR hash."""
  sig = signature(choices)
  if sig in results:
    return results[sig]
  defaults = space.default_choices()
  n_nondefault = sum(1 for k, c in choices.items() if defaults.get(k) != c)
  outcome, err = run_benchmark(cmd, choices, space, cpus=args.cpus)
  if err:
    print(f"  {label}: FAILED ({err.splitlines()[-1] if err else ''})")
    return None
  times, ir_hash = outcome
  # If an identical schedule was already measured, reuse its time.
  for row in results.values():
    if row["ir_hash"] and row["ir_hash"] == ir_hash:
      times = json.loads(row["times_json"])
      break
  total = sum(times.values())
  row = {
      "signature": sig,
      "ir_hash": ir_hash,
      "time_ns": f"{total:.1f}",
      "n_nondefault": n_nondefault,
      "times_json": json.dumps(times),
      "choices_json": json.dumps(
          {f"{p}/{k}": c for (p, k), c in sorted(choices.items())}),
  }
  append_result(out_path, row, write_header=not results)
  results[sig] = row
  print(f"  {label}: {total / 1e6:.3f} ms  (ir {ir_hash[:8]})")
  return row


def cmd_record(args):
  with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ)
    env["YNN_SCHED_TUNE"] = f"record:{args.output}"
    result = subprocess.run(list(args.command), env=env)
    if result.returncode != 0:
      sys.exit(f"benchmark failed with exit code {result.returncode}")
  space = Space(args.output)
  n_decisions = sum(len(d) for d in space.pipelines.values())
  print(f"Recorded {len(space.pipelines)} pipeline(s), {n_decisions} "
        f"decision(s), search space size {space.size():.3g}")


def cmd_random(args):
  space = Space(args.space)
  results = load_results(args.output)
  rng = random.Random(args.seed)
  defaults = space.default_choices()
  print(f"Search space size {space.size():.3g}; sampling {args.samples}")
  baseline = measure(args.command, defaults, space, args, results, args.output,
                     "baseline (defaults)")
  for i in range(args.samples):
    choices = {}
    for pipeline, d in space.decisions():
      if rng.random() < args.mutate_prob:
        choices[(pipeline, d.key)] = rng.randrange(d.num_options)
      else:
        choices[(pipeline, d.key)] = d.choice
    measure(args.command, choices, space, args, results, args.output,
            f"sample {i + 1}/{args.samples}")
  report(results, baseline)


def cmd_hillclimb(args):
  """Greedy local search: single-decision mutations from the best known."""
  space = Space(args.space)
  results = load_results(args.output)
  defaults = space.default_choices()
  rng = random.Random(args.seed)
  baseline = measure(args.command, defaults, space, args, results, args.output,
                     "baseline (defaults)")

  def choices_of(row):
    parsed = json.loads(row["choices_json"])
    return {(p, k): c
            for key, c in parsed.items()
            for p, k in [tuple(key.split("/", 1))]}

  best = min((r for r in results.values()), key=lambda r: float(r["time_ns"]))
  current = choices_of(best)
  current_time = float(best["time_ns"])
  print(f"starting from {current_time / 1e6:.3f} ms")
  stale = 0
  decisions = list(space.decisions())
  for step in range(args.steps):
    pipeline, d = rng.choice(decisions)
    option = rng.choice([o for o in range(d.num_options)
                         if o != current.get((pipeline, d.key), d.choice)])
    trial = dict(current)
    trial[(pipeline, d.key)] = option
    row = measure(args.command, trial, space, args, results, args.output,
                  f"step {step + 1}: {d.key}={option}")
    if row and float(row["time_ns"]) < current_time:
      current, current_time = trial, float(row["time_ns"])
      print(f"  -> accepted, now {current_time / 1e6:.3f} ms")
      stale = 0
    else:
      stale += 1
      if stale >= args.patience:
        print(f"no improvement in {args.patience} steps, stopping")
        break
  report(results, baseline)


def cmd_sweep(args):
  space = Space(args.space)
  results = load_results(args.output)
  defaults = space.default_choices()
  baseline = measure(args.command, defaults, space, args, results, args.output,
                     "baseline (defaults)")
  for pipeline, d in space.decisions():
    for option in range(d.num_options):
      if option == d.choice:
        continue
      choices = dict(defaults)
      choices[(pipeline, d.key)] = option
      measure(args.command, choices, space, args, results, args.output,
              f"{d.key}={option}")
  report(results, baseline)


def cmd_report(args):
  results = load_results(args.results)
  report(results, None)


def report(results, baseline):
  rows = [r for r in results.values() if r.get("time_ns")]
  if not rows:
    print("no results")
    return
  rows.sort(key=lambda r: float(r["time_ns"]))
  if baseline is None:
    defaults = [r for r in rows if r.get("n_nondefault") == "0"]
    baseline = defaults[0] if defaults else None
  base_time = float(baseline["time_ns"]) if baseline else float(
      rows[0]["time_ns"])
  distinct = len({r["ir_hash"] for r in rows if r["ir_hash"]})
  print(f"\n{len(rows)} samples, {distinct} distinct schedules; "
        f"baseline {base_time / 1e6:.3f} ms")
  base_choices = json.loads(baseline["choices_json"]) if baseline else {}
  print("top 10:")
  for r in rows[:10]:
    t = float(r["time_ns"])
    diff = {
        k: v for k, v in json.loads(r["choices_json"]).items()
        if base_choices.get(k) != v
    }
    delta = ", ".join(f"{k.split('/', 1)[-1]}={v}" for k, v in diff.items())
    print(f"  {t / 1e6:9.3f} ms  {base_time / t:5.2f}x  {delta or '(default)'}")


def cmd_stress(args):
  failures = []
  for seed in range(args.seeds):
    env = dict(os.environ)
    env["YNN_SCHED_TUNE"] = f"random:{seed}"
    result = subprocess.run(list(args.command), env=env, capture_output=True,
                            text=True)
    status = "ok" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"seed {seed}: {status}")
    if result.returncode != 0:
      failures.append(seed)
  if failures:
    sys.exit(f"failing seeds: {failures}")
  print("all seeds passed")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="mode", required=True)

  def add_command(p):
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="benchmark command after --")

  p = sub.add_parser("record", help="record the search space")
  p.add_argument("-o", "--output", required=True)
  add_command(p)
  p.set_defaults(func=cmd_record)

  for name, func in [("random", cmd_random), ("sweep", cmd_sweep),
                     ("hillclimb", cmd_hillclimb)]:
    p = sub.add_parser(name)
    p.add_argument("--space", required=True)
    p.add_argument("-o", "--output", required=True, help="results CSV")
    p.add_argument("--cpus", help="taskset CPU list, e.g. 0-7")
    if name == "random":
      p.add_argument("--samples", type=int, default=100)
      p.add_argument("--seed", type=int, default=0)
      p.add_argument("--mutate-prob", type=float, default=0.3,
                     help="probability each decision deviates from default")
    if name == "hillclimb":
      p.add_argument("--steps", type=int, default=100)
      p.add_argument("--seed", type=int, default=0)
      p.add_argument("--patience", type=int, default=30,
                     help="stop after this many non-improving steps")
    add_command(p)
    p.set_defaults(func=func)

  p = sub.add_parser("report")
  p.add_argument("results")
  p.set_defaults(func=cmd_report)

  p = sub.add_parser("stress", help="run a binary under random schedules")
  p.add_argument("--seeds", type=int, default=10)
  add_command(p)
  p.set_defaults(func=cmd_stress)

  args = parser.parse_args()
  if hasattr(args, "command"):
    if args.command and args.command[0] == "--":
      args.command = args.command[1:]
    if getattr(args, "func", None) is not cmd_report and not args.command:
      parser.error("missing benchmark command (after --)")
  args.func(args)


if __name__ == "__main__":
  main()
