import os
import re
import sys

# Usage: strip_tuner.py <runtime.cc with tuner> <output runtime.cc>
# (e.g. `git show vksnk/ynnpack-autotune:ynnpack/subgraph/runtime.cc > /tmp/full.cc`)
src = open(sys.argv[1]).read()


def cut(old, new, tag):
    global src
    assert old in src, tag
    src = src.replace(old, new)


# 1. Include.
cut('#include "ynnpack/subgraph/schedule_tuner.h"\n', "", "include")

# 2. Rename the stable-name helper: it is only used for debug output now.
cut(
    """// A stable name identifying `f` in autotuner decision keys: the symbol of its
// first output buffer. Buffer symbols are derived from value ids, so they are
// stable across rebuilds of the same subgraph.
std::string tuner_func_name(""",
    """// A stable name identifying `f` in debug output: the symbol of its first
// output buffer.
std::string debug_func_name(""",
    "name-def")
src = src.replace("tuner_func_name(", "debug_func_name(")

# 3. compute_workers: task_target hook.
cut(
    """  ynn::schedule_tuner* tuner = ynn::schedule_tuner::get();

  // Enough tasks to have good load balancing.
  slinky::index_t target_task_count = max_threads > 1 ? max_threads * 2 : 1;
  if (tuner && max_threads > 1) {
    // 0: 2x threads (the default), 1: 1x, 2: 4x, 3: 8x.
    static constexpr int multipliers[] = {2, 1, 4, 8};
    target_task_count =
        max_threads *
        multipliers[tuner->choose("task_target", "task_target", 4, 0)];
  }
""",
    """  // Enough tasks to have good load balancing.
  slinky::index_t target_task_count = max_threads > 1 ? max_threads * 2 : 1;
""",
    "task_target")

# 4. Adoption hook.
cut(
    """      bool adopt =
          !parallel_in_subtree[i] && tasks_above_lb < target_task_count;
      if (tuner) {
        // 0: the heuristic above, 1: force adoption, 2: force rejection.
        switch (tuner->choose(debug_func_name(globals, l.loop_id.func) + "." +
                                  globals.symbols.name(l.loop_id.var) +
                                  ".adopt",
                              "adopt", 3, 0)) {
          case 1: adopt = true; break;
          case 2: adopt = false; break;
        }
      }
      if (adopt) {""",
    """      const bool adopt =
          !parallel_in_subtree[i] && tasks_above_lb < target_task_count;
      if (adopt) {""",
    "adopt")

# 5. begin_pipeline block in schedule().
m = re.search(
    r"  // When the autotuner is active.*?tuner->begin_pipeline\(key\);\n  \}\n\n",
    src, re.S)
assert m, "begin_pipeline"
src = src[:m.start()] + src[m.end():]

# 6. Two-pass condition: no tuner any more.
cut(
    """  // pass 2 re-runs the walk applying those decisions. The tuner keeps the
  // single-pass behavior: its stateful decision stream must be consumed
  // exactly once.
""",
    """  // pass 2 re-runs the walk applying those decisions.
""",
    "two-pass-comment")
cut(
    """  const bool unfuse_two_pass =
      unfuse_rule_enabled() && !tuner && max_threads > 1;
  int unfuse_pass = 0;  // 0: single pass; 1: analysis; 2: apply decisions.""",
    """  const bool unfuse_two_pass = unfuse_rule_enabled() && max_threads > 1;
  int unfuse_pass = 0;  // 0: disabled; 1: analysis; 2: apply decisions.""",
    "two-pass-cond")

# 7. Split-scale hook.
m = re.search(
    r"      // Autotuner decision: scale this function's split steps.*?\n"
    r"      if \(tuner\) \{.*?\n      \}\n\n", src, re.S)
assert m, "scale"
src = src[:m.start()] + src[m.end():]

# 8. Step-clamp experiment.
m = re.search(
    r"      // EXPERIMENT \(YNN_STEP_CLAMP=1\).*?\n      \}\n\n", src, re.S)
assert m, "clamp"
src = src[:m.start()] + src[m.end():]

# 9. Fuse-depth decision.
cut(
    """      // Autotuner decision: cap how deep this function fuses. The default is
      // the un-fuse rule's answer (usually as deep as the splits can match);
      // 0 means compute root.
      int max_compute_at = rule_max_compute_at;
      if (tuner && !loop_nest.empty()) {
        max_compute_at =
            tuner->choose(debug_func_name(globals, &f) + ".fuse", "fuse",
                          static_cast<int>(loop_nest.size()) + 1,
                          rule_max_compute_at);
      }
      compute_at = 0;""",
    """      // The un-fuse rule caps how deep this function fuses (usually as deep
      // as the splits can match); 0 means compute root.
      const int max_compute_at = rule_max_compute_at;
      compute_at = 0;""",
    "fuse")

# 10. Inline-decision comment (tuner path is gone; only symbolic-batch left).
cut(
    """          } else {
            // Single-pass (tuner) and symbolic-batch candidates decide
            // inline with the (possibly conservative) current nest bound.""",
    """          } else {
            // Symbolic-batch candidates decide inline with the (possibly
            // conservative) current nest bound.""",
    "inline-comment")

# 11. end_pipeline.
cut(
    """
  if (tuner) {
    tuner->end_pipeline();
  }
}
""",
    """
}
""",
    "end_pipeline")

# 12. Keep the IR dump but drop the autotuner mention.
cut(
    """    // Appends each built pipeline, so a process that builds several dumps all
    // of them; the autotuner hashes the file to recognize schedules it has
    // already benchmarked.""",
    """    // Appends each built pipeline, so a process that builds several dumps
    // all of them.""",
    "ir-comment")

assert "tuner" not in src.replace("autotuner", "").replace("Autotuner", ""), \
    [l for l in src.splitlines() if "tuner" in l][:5]
open(sys.argv[2], "w").write(src)
print("stripped ok")
