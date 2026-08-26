// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_SCHEDULE_TUNER_H_
#define XNNPACK_YNNPACK_SUBGRAPH_SCHEDULE_TUNER_H_

#include <cstdint>
#include <string>

namespace ynn {

// Record/replay hook for the scheduler's discrete decisions, driven by the
// YNN_SCHED_TUNE environment variable:
//
//   YNN_SCHED_TUNE=record:<path>  every decision point is appended to <path>
//                                 with its default choice; scheduling is
//                                 unchanged. The resulting file describes the
//                                 search space for the autotuner driver.
//   YNN_SCHED_TUNE=replay:<path>  decisions listed in <path> take the choice
//                                 from the file; everything else defaults.
//   YNN_SCHED_TUNE=random:<seed>  every decision takes a per-key deterministic
//                                 pseudo-random choice; for stress testing
//                                 that the whole space produces correct
//                                 schedules (run the unit tests with this).
//
// The file has one `pipeline <key>` line per scheduled pipeline, followed by
// one line per decision: `<decision_key> <kind> <num_options> <choice>`.
// Decisions are keyed by stable names (output buffer symbols and loop
// variables), not by position, so a choice keeps its meaning when other
// choices change the schedule around it, and so that rebuilding the same
// pipeline replays identically.
class schedule_tuner {
 public:
  // The process-wide tuner, or nullptr when YNN_SCHED_TUNE is not set.
  static schedule_tuner* get();

  // Brackets the decision sequence of one pipeline. `key` identifies the
  // pipeline (a hash of its structure); decisions made between begin and end
  // belong to it. Decisions made outside a bracket are ignored (defaults).
  void begin_pipeline(uint64_t key);
  void end_pipeline();

  // One decision point with options 0..num_options-1. Returns the choice:
  // the file's (clamped to the valid range) on replay, a seeded hash of the
  // key on random, `default_choice` otherwise. Decision points with fewer
  // than two options are not recorded.
  int choose(const std::string& decision_key, const char* kind,
             int num_options, int default_choice);

 private:
  schedule_tuner() = default;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_SCHEDULE_TUNER_H_
