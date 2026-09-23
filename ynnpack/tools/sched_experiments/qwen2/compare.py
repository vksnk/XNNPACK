#!/usr/bin/env python3
"""Join threshold-rule and cost-model matrices into one comparison."""
import json, os

SP = os.path.dirname(os.path.abspath(__file__))
thr = json.load(open(f"{SP}/matrix.json"))
cost = json.load(open(f"{SP}/matrix_cost.json"))

print(f"{'case':<40} {'default':>9} {'thresh':>8} {'cost':>8} "
      f"{'cost cpu':>8}")
worst = []
for label in thr:
    t = thr[label]
    c = cost.get(label)
    if not c:
        continue
    d = min(t["t_off"], c["t_off"])
    s_thr = d / t["t_on"]
    s_cost = d / c["t_on"]
    cpu = c["c_on"] / c["c_off"]
    flag = ""
    if s_cost < 0.97:
        flag = " <-- REGRESSION"
        worst.append((s_cost, label))
    elif s_cost < s_thr - 0.08:
        flag = " (thresh better)"
    print(f"{label:<40} {d:>8.1f}u {s_thr:>7.2f}x {s_cost:>7.2f}x "
          f"{cpu:>7.2f}x{flag}")
if worst:
    print("\nregressions (cost model):")
    for s, l in sorted(worst):
        print(f"  {s:.2f}x {l}")
else:
    print("\nno cost-model regressions")
