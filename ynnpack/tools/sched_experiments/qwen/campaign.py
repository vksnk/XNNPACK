#!/usr/bin/env python3
"""Exhaustive blockwise-int4 schedule search over a Qwen3-0.6B shape grid.

Per case: record space -> default + rule candidates -> random search ->
hillclimb -> stable re-measure of top-3 -> ablation to minimal decision set.
Two worker lanes pinned to the two CCDs. Resumable: finished cases are read
from results.jsonl and skipped.
"""
import subprocess, os, json, random, time, sys, hashlib
from multiprocessing import Process

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
BENCH = "bazel-bin/ynnpack/composites/dot_bench"
RESULTS = os.path.join(SP, "results.jsonl")
DEADLINE = time.time() + float(sys.argv[1]) * 60 if len(sys.argv) > 1 else time.time() + 165 * 60

NK_PAIRS = [(1024, 1024), (2048, 1024), (1024, 2048), (3072, 1024), (1024, 3072)]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
CASES = [(m, n, k) for n, k in NK_PAIRS for m in MS]
CASES += [(1, 151936, 1024), (4, 151936, 1024)]


def case_name(m, n, k):
  return f"m{m}_n{n}_k{k}"


def parse_space(path):
  space = []
  cur = None
  for line in open(path):
    p = line.split()
    if not p:
      continue
    if p[0] == "pipeline":
      cur = p[1]
      space.append((cur, None))
    else:
      space.append((cur, (p[0], p[1], int(p[2]), int(p[3]))))
  return space


class Case:

  def __init__(self, m, n, k, cpus):
    self.m, self.n, self.k, self.cpus = m, n, k, cpus
    self.name = case_name(m, n, k)
    self.space_path = f"{SP}/{self.name}.space"
    self.replay_path = f"{SP}/{self.name}.replay"
    self.cache = {}  # signature -> time

  def bench_cmd(self, min_time):
    return ["taskset", "-c", self.cpus, BENCH, "--thread_count=4",
            f"--shape={self.m},{self.n},{self.k}",
            "--benchmark_filter=blockwise_int4_bs32",
            f"--benchmark_min_time={min_time}"]

  def record(self):
    env = dict(os.environ, YNN_SCHED_TUNE=f"record:{self.space_path}")
    subprocess.run(self.bench_cmd("1x"), env=env, capture_output=True)
    self.space = parse_space(self.space_path)
    self.defaults = {f"{pk}/{d[0]}": d[3] for pk, d in self.space if d}
    self.decisions = [(pk, d) for pk, d in self.space if d]

  def measure(self, choices, min_time="0.25s", repeats=1, cache=True):
    """choices: {pipeline/key: choice}; returns median time in ns or None."""
    sig = json.dumps(sorted(choices.items())) + min_time + str(repeats)
    sig = hashlib.sha256(sig.encode()).hexdigest()
    if cache and sig in self.cache:
      return self.cache[sig]
    lines = []
    for pk, d in self.space:
      if d is None:
        lines.append(f"pipeline {pk}")
      else:
        c = choices.get(f"{pk}/{d[0]}", d[3])
        lines.append(f"{d[0]} {d[1]} {d[2]} {c}")
    with open(self.replay_path, "w") as f:
      f.write("\n".join(lines) + "\n")
    env = dict(os.environ, YNN_SCHED_TUNE=f"replay:{self.replay_path}")
    times = []
    for _ in range(repeats):
      r = subprocess.run(self.bench_cmd(min_time), env=env,
                         capture_output=True, text=True)
      if "error" in r.stdout.lower() or "error" in r.stderr.lower():
        return None
      for line in r.stdout.splitlines():
        if "real_time" in line:
          times.append(float(line.split()[1]))
          break
      else:
        return None
    times.sort()
    t = times[len(times) // 2]
    if cache:
      self.cache[sig] = t
    return t

  def key(self, name):
    for pk, d in self.space:
      if d and d[0] == name:
        return f"{pk}/{name}", d[2]
    return None, 0

  def rule_candidates(self):
    """Hand-crafted candidate schedules derived from earlier findings."""
    cands = {}
    fuse_key, fuse_n = self.key("v17.fuse")
    d2_key, _ = self.key("v17.d2.scale")
    out3d1, _ = self.key("out3.d1.scale")
    out3d0, _ = self.key("out3.d0.scale")
    if fuse_key:
      cands["fuse0"] = {fuse_key: 0}
      cands["fuse1"] = {fuse_key: 1}
      if fuse_n > 3:
        cands["fuse2"] = {fuse_key: 2}
      if out3d1:
        cands["fuse0_out3d1div4"] = {fuse_key: 0, out3d1: 4}
      elif out3d0:
        cands["fuse0_out3d0div4"] = {fuse_key: 0, out3d0: 4}
      if d2_key:
        cands["fuse1_d2div2"] = {fuse_key: 1, d2_key: 3}
        cands["fuse1_d2div4"] = {fuse_key: 1, d2_key: 4}
    return cands

  def run(self, rng, samples=150, hc_steps=80):
    self.record()
    result = {"m": self.m, "n": self.n, "k": self.k}
    base = self.measure({}, repeats=3)
    if base is None:
      result["error"] = "default failed"
      return result
    result["default_ns"] = base

    evaluated = {}  # frozen choices json -> (time, choices)

    def consider(label, choices):
      t = self.measure(choices)
      if t is not None:
        evaluated[json.dumps(sorted(choices.items()))] = (t, choices, label)
      return t

    cands = self.rule_candidates()
    result["candidates"] = {}
    for label, ch in cands.items():
      t = consider(label, ch)
      result["candidates"][label] = t

    # Random search.
    for i in range(samples):
      if time.time() > DEADLINE:
        break
      choices = {}
      for pk, d in self.decisions:
        if rng.random() < 0.35:
          choices[f"{pk}/{d[0]}"] = rng.randrange(d[2])
      consider("random", choices)

    best_t, best_c, _ = min(evaluated.values(), key=lambda v: v[0])

    # Hillclimb from the best.
    stale = 0
    for i in range(hc_steps):
      if time.time() > DEADLINE or stale > 30:
        break
      pk, d = rng.choice(self.decisions)
      key = f"{pk}/{d[0]}"
      cur = best_c.get(key, d[3])
      opt = rng.choice([o for o in range(d[2]) if o != cur])
      trial = dict(best_c)
      trial[key] = opt
      t = consider("hc", trial)
      if t is not None and t < best_t:
        best_t, best_c = t, trial
        stale = 0
      else:
        stale += 1

    # Stable re-measure of the top 3 distinct candidates.
    top = sorted(evaluated.values(), key=lambda v: v[0])[:3]
    stable = []
    for t, c, label in top:
      st = self.measure(c, min_time="0.5s", repeats=3, cache=False)
      if st is not None:
        stable.append((st, c, label))
    stable.sort(key=lambda v: v[0])
    best_t, best_c, best_label = stable[0]
    base = self.measure({}, min_time="0.5s", repeats=3, cache=False) or base
    result["default_ns"] = base
    result["best_ns"] = best_t
    result["speedup"] = base / best_t
    result["best_label"] = best_label

    # Ablate to the minimal decision set.
    nondef = {k: v for k, v in best_c.items() if self.defaults.get(k) != v}
    keep = []
    for k in sorted(nondef):
      t = self.measure({kk: vv for kk, vv in best_c.items() if kk != k})
      if t is None or t > best_t * 1.05:
        keep.append(k)
    minimal = {k: nondef[k] for k in keep}
    tmin = self.measure(minimal, min_time="0.5s", repeats=3, cache=False)
    if tmin is not None and tmin <= best_t * 1.05:
      result["minimal"] = {k.split("/", 1)[1]: v for k, v in minimal.items()}
      result["minimal_ns"] = tmin
    else:
      result["minimal"] = {k.split("/", 1)[1]: v for k, v in nondef.items()}
      result["minimal_ns"] = best_t
    result["best_choices"] = {k.split("/", 1)[1]: v for k, v in nondef.items()}
    return result


def lane(cases, cpus, lane_id):
  os.chdir(ROOT)
  rng = random.Random(1234 + lane_id)
  done = set()
  if os.path.exists(RESULTS):
    for line in open(RESULTS):
      try:
        r = json.loads(line)
        done.add(case_name(r["m"], r["n"], r["k"]))
      except Exception:
        pass
  for m, n, k in cases:
    if case_name(m, n, k) in done:
      continue
    if time.time() > DEADLINE:
      print(f"[lane{lane_id}] deadline reached", flush=True)
      break
    t0 = time.time()
    try:
      result = Case(m, n, k, cpus).run(rng)
    except Exception as e:
      result = {"m": m, "n": n, "k": k, "error": repr(e)}
    result["wall_s"] = round(time.time() - t0, 1)
    with open(RESULTS, "a") as f:
      f.write(json.dumps(result) + "\n")
    print(f"[lane{lane_id}] {case_name(m,n,k)}: "
          f"{result.get('speedup', 0):.2f}x in {result['wall_s']}s", flush=True)


if __name__ == "__main__":
  lanes = [CASES[0::2], CASES[1::2]]
  ps = [Process(target=lane, args=(lanes[0], "0-5", 0)),
        Process(target=lane, args=(lanes[1], "6-11", 1))]
  for p in ps:
    p.start()
  for p in ps:
    p.join()
  print("CAMPAIGN COMPLETE", flush=True)
