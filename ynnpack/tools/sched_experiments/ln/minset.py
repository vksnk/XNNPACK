#!/usr/bin/env python3
"""Mask 3 minimal-set candidates vs full winner."""
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "validate", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "validate.py"))
v = importlib.util.module_from_spec(spec)
import sys
sys.argv = ["minset"]
# Avoid running validate.py's __main__ block.
src = open(spec.origin).read().split('if __name__ == "__main__":')[0]
exec(compile(src, spec.origin, "exec"), v.__dict__)

pipe = [e[1] for e in v.load_space(3) if e[0] == "pipeline"][0]


def c(**kw):
    return {f"{pipe}/{k.replace('_', '.')}": val for k, val in kw.items()}


candidates = [
    ("v10 only", c(**{"v10.d0.scale".replace('.', '_'): 4})),
]
# build explicitly to avoid replace confusion
candidates = [
    ("v10/4", {f"{pipe}/v10.d0.scale": 4}),
    ("v10/4+v2/4", {f"{pipe}/v10.d0.scale": 4, f"{pipe}/v2.d0.scale": 4}),
    ("v10/4+v2/4+v3/4", {f"{pipe}/v10.d0.scale": 4, f"{pipe}/v2.d0.scale": 4,
                          f"{pipe}/v3.d0.scale": 4}),
    ("winner", v.winner(3)),
]
for name, ch in candidates:
    td, tw, cd, cw = v.ab(3, ch, rounds=2)
    print(f"{name:<18} default {td:7.1f}u -> {tw:7.1f}u  {td/tw:5.2f}x  "
          f"cpu {cw/cd:4.2f}x", flush=True)
print("MINSET DONE", flush=True)
