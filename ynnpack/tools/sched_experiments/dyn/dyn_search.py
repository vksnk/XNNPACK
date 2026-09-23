#!/usr/bin/env python3
"""Search blockwise QB4W schedules on the DYNAMIC-shape delegate graph.

The graph is shape-symbolic (m dynamic), so one decision space serves every
runtime M. Search independently at several M values, then cross-replay each
winner at all Ms to test m-invariance of the winning schedules.
"""
import subprocess, os, time, json, random
from multiprocessing import Process

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Projects/XNNPACK")
SPACE = f"{SP}/m1.space"
MS = [1, 16, 64, 256]

space = []
cur = None
for line in open(SPACE):
    p = line.split()
    if p[0] == "pipeline":
        cur = p[1]
        space.append((cur, None))
    else:
        space.append((cur, (p[0], p[1], int(p[2]), int(p[3]))))
DECISIONS = [(pk, d) for pk, d in space if d]


def write_replay(path, choices):
    lines = []
    for pk, d in space:
        if d is None:
            lines.append(f"pipeline {pk}")
        else:
            lines.append(f"{d[0]} {d[1]} {d[2]} {choices.get(f'{pk}/{d[0]}', d[3])}")
    open(path, "w").write("\n".join(lines) + "\n")


def run(m, choices, cpus, lane, min_time="0.35s"):
    replay = f"{SP}/replay{lane}.txt"
    write_replay(replay, choices)
    env = dict(os.environ, YNN_SCHED_TUNE=f"replay:{replay}")
    r = subprocess.run(
        ["taskset", "-c", cpus, "bazel-bin/bench/subgraph/fully_connected",
         f"--benchmark_filter=QD8F32QB4WFullyConnected/M:{m}/K:2048/N:512/",
         "--num_threads=4", f"--benchmark_min_time={min_time}"],
        env=env, capture_output=True, text=True, cwd=ROOT)
    for line in r.stdout.splitlines():
        if "real_time" in line:
            return float(line.split()[1])
    return None


def canary(cpus, lane):
    while True:
        t = run(16, {}, cpus, lane)
        if t is not None and t < 60:  # healthy M:16 default ~= 40us
            return
        print(f"  (canary {t}, waiting)", flush=True)
        time.sleep(30)


def search(m, cpus, lane):
    rng = random.Random(100 + m)
    out = f"{SP}/search_m{m}.jsonl"
    evaluated = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            evaluated[json.dumps(sorted(r["choices"].items()))] = (r["t"], r["choices"])

    def consider(choices):
        key = json.dumps(sorted(choices.items()))
        if key in evaluated:
            return evaluated[key][0]
        t = run(m, choices, cpus, lane)
        if t is not None:
            evaluated[key] = (t, choices)
            with open(out, "a") as f:
                f.write(json.dumps({"t": t, "choices": choices}) + "\n")
        return t

    canary(cpus, lane)
    base = consider({})
    for i in range(130):
        if i % 25 == 24:
            canary(cpus, lane)
        choices = {}
        for pk, d in DECISIONS:
            if rng.random() < 0.3:
                choices[f"{pk}/{d[0]}"] = rng.randrange(d[2])
        consider(choices)
    best_t, best_c = min(evaluated.values(), key=lambda v: v[0])
    stale = 0
    for i in range(60):
        if stale > 30:
            break
        if i % 25 == 24:
            canary(cpus, lane)
        pk, d = rng.choice(DECISIONS)
        key = f"{pk}/{d[0]}"
        curv = best_c.get(key, d[3])
        opt = rng.choice([o for o in range(d[2]) if o != curv])
        trial = dict(best_c)
        trial[key] = opt
        t = consider(trial)
        if t is not None and t < best_t:
            best_t, best_c = t, trial
            stale = 0
        else:
            stale += 1
    print(f"[m={m}] default={base:.1f}us best={best_t:.1f}us ({base/best_t:.2f}x)",
          flush=True)
    json.dump({"m": m, "default": base, "best": best_t, "choices": best_c},
              open(f"{SP}/best_m{m}.json", "w"))


def lane_fn(ms, cpus, lane):
    for m in ms:
        if os.path.exists(f"{SP}/best_m{m}.json"):
            continue
        search(m, cpus, lane)


if __name__ == "__main__":
    ps = [Process(target=lane_fn, args=([1, 64], "0-5", 0)),
          Process(target=lane_fn, args=([16, 256], "6-11", 1))]
    for p in ps: p.start()
    for p in ps: p.join()

    # Cross matrix: default + each M's winner, evaluated at every M.
    print("\ncross matrix (rows=schedule, cols=runtime M):", flush=True)
    schedules = [("default", {})]
    for m in MS:
        b = json.load(open(f"{SP}/best_m{m}.json"))
        schedules.append((f"win_m{m}", b["choices"]))
    header = "schedule    " + "".join(f"{('M='+str(m)):>12}" for m in MS)
    print(header, flush=True)
    results = {}
    for name, ch in schedules:
        row = []
        for m in MS:
            canary("6-11", 9)
            t = min(x for x in (run(m, ch, "6-11", 9, "0.5s"),
                                run(m, ch, "6-11", 9, "0.5s")) if x)
            results[(name, m)] = t
            row.append(t)
        print(f"{name:<12}" + "".join(f"{t:>11.1f}u" for t in row), flush=True)
    json.dump({f"{n}@{m}": t for (n, m), t in results.items()},
              open(f"{SP}/cross.json", "w"))
    open(f"{SP}/DONE", "w").write("done\n")
    print("DYN SEARCH DONE", flush=True)
