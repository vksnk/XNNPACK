// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/runtime.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifdef YNN_ENABLE_PERFETTO
#include "ynnpack/subgraph/perfetto.h"
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
#include "xla/tsl/profiler/lib/traceme.h"
#endif
#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/ref_count.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/schedule_tuner.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/tensor.h"
#include "slinky/base/arithmetic.h"
#include "slinky/base/thread_pool.h"
#include "slinky/builder/node_mutator.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/depends_on.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/print.h"
#include "slinky/runtime/stmt.h"

void ynn_runtime_value::make_buffer(ynn_runtime& runtime,
                                    slinky::expr elem_size) {
  if (buffer) {
    assert(buffer->sym() == symbol);
    return;
  }
  if (!symbol.defined()) {
    symbol = runtime.globals.symbols.insert_unique(name());
  }
  buffer = ynn::make_buffer_expr(symbol, rank(), std::move(elem_size));
  for (size_t i = 0; i < rank(); ++i) {
    if (!extents[i].defined() || slinky::is_constant(extents[i], 1)) {
      buffer->dim(i) = slinky::dim::broadcast();
    }
  }
}

void ynn_runtime_value::make_buffer(ynn_runtime& runtime) {
  make_buffer(runtime, ynn::type_size_bytes(type));
}

std::unique_ptr<ynn::scheduling_info> ynn_runtime::make_schedule(
    ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
    const slinky::expr& element_cost,
    ynn::span<const slinky::expr> given_splits,
    ynn::span<const int> loop_order) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }

  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, element_cost, given_splits, loop_order);

  return make_schedule(dims, extents, splits, loop_order);
}

std::unique_ptr<ynn::scheduling_info> ynn_runtime::make_schedule(
    ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
    ynn::span<const slinky::expr> splits, ynn::span<const int> loop_order) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }

  auto get_loop_dim = [&](int index_d) {
    return index_d < loop_order.size() ? loop_order[index_d] : index_d;
  };

  std::vector<ynn::scheduling_split> loop_splits;
  for (int index_d = 0; index_d < rank; ++index_d) {
    int d = get_loop_dim(index_d);
    if (extents[d].defined() && splits[d].defined()) {
      loop_splits.push_back({dims[d], splits[d], extents[d]});
    }
  }

  auto scheduling_info = std::make_unique<ynn::scheduling_info>();
  scheduling_info->loop_splits = std::move(loop_splits);
  return scheduling_info;
}

namespace {

// Just a helper structure to track information about loop levels.
struct loop_level {
  slinky::loop_id loop_id;
  slinky::expr extent;
  slinky::expr step;
  bool step_is_required = false;
  // Any step installed on this loop must be a multiple of this value, see
  // scheduling_split::step_alignment.
  slinky::index_t step_alignment = 1;
  // A constant step deliberately pinned by the un-fuse rule to bound the
  // materialized tile for any runtime shape. Reconciliation must not replace
  // it: a producer's required step is a cache-blocking preference (kernels
  // handle partial tiles), and an lcm with a symbolic split would grow with
  // the batch again.
  bool step_pinned = false;
  // The index of the parent loop in the global loop nest, or -1 for the
  // outermost loops. Loops are appended after their parent, so the parent
  // index is always less than the index of the loop itself.
  int parent = -1;
  // The number of workers the loop should use, computed by compute_workers()
  // once the whole nest is built.
  slinky::expr workers = slinky::loop::serial;
  // A finer step for this loop, proposed by a function fused into it. Set by
  // reconcile_step(); compute_workers() decides whether to use it, because
  // that depends on the rest of the loop nest, which isn't built yet when
  // functions are matched.
  slinky::expr proposed_step;
};

struct scheduling_data {
  // Loop nest should be a pair of function and loop_id. This is in fact a
  // tree, but for the sake of simplicity we store it as a set of pathes
  // (loop nests in this case) from the root of the tree (outermost
  // location) to a leaf node (the innermost loop of a given function). Loop
  // nests for each of the functions scheduled so far with the indices
  // pointing to the global loop nest. Loop nests can overlap with each
  // other if functions are scheduled within the same loop (only prefixes,
  // i.e. from the root of the loop nest to the most innermost common loop).
  std::vector<int> loop_nest;
  // Compute_at locations of a function -- this an index within a
  // loop nest of a given function, i.e from root (0) to the innermost loop
  // (loop_nest[f_index].back()).
  int compute_at = 0;
  // For each split of the function's scheduling_info (in the reversed,
  // outermost-first order used during matching), whether it was matched to
  // an existing loop of the nest. Matched splits become loops of the
  // function this function was fused into; unmatched splits become the
  // function's own loops.
  std::vector<bool> split_matched;
};

template <typename T, typename Target>
bool find_n(const T* data, size_t size, const Target& x) {
  for (size_t i = 0; i < size; ++i) {
    if (data[i] == x) {
      return true;
    }
  }
  return false;
}

// Finds which {`buffer`, `dim`} corresponds to the output dimension variable
// `v`.
std::pair<slinky::var, int> find_output_dim(const slinky::func* f,
                                            const slinky::var& v) {
  if (f) {
    for (const auto& out : f->outputs()) {
      for (int i = 0; i < out.dims.size(); ++i) {
        if (out.dims[i] == v) {
          return {out.sym(), i};
        }
      }
    }
  }
  return {slinky::var(), -1};
}

// Least common multiple of two (possibly symbolic) loop steps, evaluated at
// runtime. Used to reconcile two producers that require different tiles for a
// shared loop: a multiple of both keeps the fused loop an integer number of
// each producer's tile (and hence a multiple of each kernel's m/n block).
// If the LCM overflows, it clamps at the max index_t value.
slinky::expr lcm_sat(ynn::slinky_globals& globals, slinky::expr a,
                     slinky::expr b) {
  if (slinky::prove_true(a == b, globals.fact_bounds, globals.fact_alignment)) {
    return a;
  }
  auto impl = [](const slinky::call* op,
                 slinky::eval_context& ctx) -> slinky::index_t {
    slinky::index_t a_val = slinky::evaluate(op->args[0], ctx);
    slinky::index_t b_val = slinky::evaluate(op->args[1], ctx);
    assert(a_val != 0 && b_val != 0);
    return slinky::mul_sat(a_val / slinky::gcd(a_val, b_val), b_val);
  };
  return globals.get(slinky::call::make(impl, {std::move(a), std::move(b)}),
                     "lcm_sat");
}

// This pass infers symbolic source regions for all buffers, traversing the
// pipeline in reverse topological order (consumers to producers). It ensures
// that loops are only fused if their dimensions share a common source
// region origin, which naturally prevents incorrect fusions of unrelated
// dimensions that happen to have the same constant size. This is similar
// to the backward bounds inference, but much more lightweight because we
// only care if given extents are the same in terms of consumer extents.
std::map<std::pair<slinky::var, int>, int> infer_source_regions(
    const std::vector<slinky::func>& funcs) {
  // Maps {buffer_sym, dim_index} to its inferred source region unique
  // identifier.
  std::map<std::pair<slinky::var, int>, int> source_regions;

  int next_source_region_id = 0;

  // Lazily creates a new symbolic source region identifier if one doesn't
  // exist.
  auto get_source_region = [&](slinky::var buf, int dim) {
    auto key = std::make_pair(buf, dim);
    if (source_regions.find(key) == source_regions.end()) {
      source_regions[key] = next_source_region_id++;
    }
    return source_regions[key];
  };

  // Traverses operations backwards to propagate source region symbols.
  for (int i = funcs.size() - 1; i >= 0; --i) {
    const slinky::func& f = funcs[i];
    if (f.outputs().empty()) continue;

    // Collect all unique output variables for this function.
    std::set<slinky::var> out_vars;
    for (const auto& out : f.outputs()) {
      for (auto v : out.dims) {
        out_vars.insert(v);
      }
    }

    const auto* sched = static_cast<const ynn::scheduling_info*>(f.user_data());

    for (size_t in_idx = 0; in_idx < f.inputs().size(); ++in_idx) {
      const auto& in = f.inputs()[in_idx];
      const std::vector<slinky::interval_expr>* scheduler_bounds = nullptr;
      if (sched && in_idx < sched->input_scheduler_bounds.size()) {
        scheduler_bounds = &sched->input_scheduler_bounds[in_idx];
      }

      for (int d = 0; d < in.bounds.size(); ++d) {
        slinky::interval_expr bound = in.bounds[d];

        // If this function provided a custom scheduler bound for this input
        // dimension, we use it instead of the real bounds. This allows tricks
        // like fusing pack_b's blocks_n (extent N/16) with dot's n (extent N)
        // by declaring a virtual 1-to-1 mapping. The bounds are attached to
        // the consumer that understands the layout of the input, so they work
        // regardless of which function produced the input.
        if (scheduler_bounds && d < scheduler_bounds->size() &&
            (*scheduler_bounds)[d].min.defined()) {
          bound = (*scheduler_bounds)[d];
        }

        // Find which output variables this input dimension depends on.
        slinky::var correlated_var;
        for (auto v : out_vars) {
          if (slinky::depends_on(bound, v).any()) {
            if (correlated_var.defined()) {
              correlated_var = slinky::var();
              break;
            }
            correlated_var = v;
          }
        }

        int inferred_region = next_source_region_id++;

        if (correlated_var.defined()) {
          slinky::var v = correlated_var;

          // Collect source regions for this variable from all outputs that
          // contain it.
          std::vector<int> parent_source_regions;
          for (const auto& out : f.outputs()) {
            for (int od = 0; od < out.dims.size(); ++od) {
              if (out.dims[od] == v) {
                parent_source_regions.push_back(
                    get_source_region(out.sym(), od));
              }
            }
          }

          if (slinky::is_variable(bound.min, v) &&
              slinky::is_variable(bound.max, v) &&
              !parent_source_regions.empty()) {
            // Check if all parent extents are equivalent.
            bool all_equal = true;
            for (size_t k = 1; k < parent_source_regions.size(); ++k) {
              if (parent_source_regions[0] != parent_source_regions[k]) {
                all_equal = false;
                break;
              }
            }
            if (all_equal) {
              inferred_region = parent_source_regions[0];
            }
          }
        }

        auto key = std::make_pair(in.sym(), d);
        if (source_regions.count(key) > 0) {
          // If this buffer has multiple consumers with conflicting inferred
          // regions, merge them into a new unique ID (breaks fusion for this
          // dimension).
          if (source_regions[key] != inferred_region) {
            source_regions[key] = next_source_region_id++;
          }
        } else {
          source_regions[key] = inferred_region;
        }
      }
    }
  }

  return source_regions;
}

// If `e` is a global let variable, return the let's value (an existing,
// shared expression -- nothing is constructed); otherwise return `e`
// unchanged. One level is enough for the analyses below: loop steps are let
// variables whose immediate values expose the min/max caps the constant
// bounds need (variables remaining inside them are simply unknowns to the
// bounds evaluator), and loop extents are stored in raw form. This
// deliberately avoids substituting lets into expressions: full expansion
// can grow combinatorially when let values nest.
slinky::expr resolve_let_var(const ynn::slinky_globals& globals,
                             const slinky::expr& e) {
  if (auto v = slinky::as_variable(e)) {
    for (const auto& let : globals.lets) {
      if (let.first == *v) return let.second;
    }
  }
  return e;
}

// The condition under which `l` runs exactly one full-extent iteration, i.e.
// whoever created it made no real splitting decision for this dimension:
// make_split_factors() hands out a cache-sized tile area and returns the whole
// extent for any dimension that fits in what is left of it. This is the
// weakest claim a function can make on a loop step. Such a loop can be folded
// away entirely, it can take a finer step from a function fused into it, and
// it cannot be the loop that parallelizes the nest.
//
// The step is resolved one let level deep and the extent simplified before
// comparing: split factors are usually `min(...)` expressions the simplifier
// already reduced to the extent itself, but hidden behind a let variable the
// prover can't see through.
slinky::expr is_single_iteration(const ynn::slinky_globals& globals,
                                 const loop_level& l) {
  if (!l.extent.defined() || !l.step.defined()) return slinky::expr();
  return slinky::simplify(l.extent) <= resolve_let_var(globals, l.step);
}

// Round `step` up to a multiple of `alignment`.
slinky::expr align_step(slinky::expr step, slinky::index_t alignment) {
  if (alignment <= 1 || !step.defined()) return step;
  return ((step + (alignment - 1)) / alignment) * alignment;
}

// A stable name identifying `f` in autotuner decision keys: the symbol of its
// first output buffer. Buffer symbols are derived from value ids, so they are
// stable across rebuilds of the same subgraph.
std::string tuner_func_name(const ynn::slinky_globals& globals,
                            const slinky::func* f) {
  if (f && !f->outputs().empty()) {
    return globals.symbols.name(f->outputs()[0].sym());
  }
  return "?";
}

// A guaranteed constant lower bound on the number of iterations of a loop
// with the given extent and step, or 1 when either bound is unknown (a
// scheduled loop runs at least one iteration). Steps are often global let
// variables whose values reference the opaque dot split factors, so both are
// run through bounds_of with the registered facts before asking for a
// constant bound.
slinky::index_t iterations_lower_bound(ynn::slinky_globals& globals,
                                       const slinky::expr& extent,
                                       const slinky::expr& step) {
  std::optional<slinky::index_t> extent_lb =
      slinky::evaluate_constant_lower_bound(
          slinky::bounds_of(extent, globals.fact_bounds, globals.fact_alignment)
              .min);
  std::optional<slinky::index_t> step_ub =
      slinky::evaluate_constant_upper_bound(
          slinky::bounds_of(resolve_let_var(globals, step), globals.fact_bounds,
                            globals.fact_alignment)
              .max);
  if (extent_lb && step_ub && *step_ub > 0) {
    return std::max<slinky::index_t>(slinky::ceil_div(*extent_lb, *step_ub), 1);
  }
  return 1;
}

// A guaranteed lower bound on the number of parallel tasks the given levels
// of the loop nest produce: the product of each pure loop's provable
// iteration count. Reduction loops run their iterations within one task, so
// they contribute nothing.
slinky::index_t nest_tasks_lower_bound(ynn::slinky_globals& globals,
                                       std::vector<loop_level>& nest,
                                       const std::vector<int>& levels) {
  slinky::index_t tasks = 1;
  for (int i : levels) {
    const loop_level& l = nest[i];
    if (!globals.is_pure_dim(l.loop_id.var)) continue;
    tasks = slinky::mul_sat(tasks,
                            iterations_lower_bound(globals, l.extent, l.step));
  }
  return tasks;
}

// The same lower bound for the loops a function would run itself if computed
// at root.
slinky::index_t split_tasks_lower_bound(
    ynn::slinky_globals& globals,
    const std::vector<ynn::scheduling_split>& splits) {
  slinky::index_t tasks = 1;
  for (const ynn::scheduling_split& s : splits) {
    if (!globals.is_pure_dim(s.var)) continue;
    tasks = slinky::mul_sat(tasks,
                            iterations_lower_bound(globals, s.extent, s.step));
  }
  return tasks;
}

// A provable constant upper bound of the bytes of `f`'s first output, i.e.
// what computing `f` at root would materialize. The extents come from the
// function's declared splits (which span its iteration space); the buffer's
// own dim bounds are frequently behind expressions the constant evaluator
// can't resolve. nullopt when any extent has no constant bound (e.g. dynamic
// shapes).
std::optional<slinky::index_t> output_bytes_upper_bound(
    const slinky::func& f,
    const std::vector<ynn::scheduling_split>& loop_splits) {
  if (f.outputs().empty()) return std::nullopt;
  std::optional<slinky::index_t> elem =
      slinky::as_constant(f.outputs()[0].buffer->elem_size());
  if (!elem) return std::nullopt;
  slinky::index_t total = *elem;
  for (const ynn::scheduling_split& s : loop_splits) {
    std::optional<slinky::index_t> ub =
        slinky::evaluate_constant_upper_bound(s.extent);
    if (!ub) return std::nullopt;
    total = slinky::mul_sat(total, std::max<slinky::index_t>(*ub, 1));
  }
  return total;
}

// The un-fuse rule below trades memory for parallelism, so it is capped by
// roughly the size of a shared last-level cache. Overridable for experiments
// via YNN_UNFUSE_BUDGET (bytes); YNN_UNFUSE=0 disables the rule entirely.
bool unfuse_rule_enabled() {
  static const bool enabled = [] {
    const char* v = getenv("YNN_UNFUSE");
    return !v || atoi(v) != 0;
  }();
  return enabled;
}

slinky::index_t unfuse_bytes_budget() {
  static const slinky::index_t budget = [] {
    const char* v = getenv("YNN_UNFUSE_BUDGET");
    return v ? static_cast<slinky::index_t>(atoll(v))
             : static_cast<slinky::index_t>(24) << 20;
  }();
  return budget;
}

// The constant batch-tile step used by the symbolic-batch arm of the un-fuse
// rule (YNN_UNFUSE_TILE to override).
slinky::index_t unfuse_batch_tile() {
  static const slinky::index_t tile = [] {
    const char* v = getenv("YNN_UNFUSE_TILE");
    return v ? static_cast<slinky::index_t>(atoll(v))
             : static_cast<slinky::index_t>(32);
  }();
  return tile;
}

// Decide how many workers each loop of the global nest should use. This must
// run after the whole nest is built (and all the steps are final): after
// fusion, a function's loops can end up inside loops of other functions, so
// the number of tasks produced outside each loop is only known once the nest
// is complete.
//
// `funcs_in_level[i]` is the number of functions whose body executes inside
// loop level `i`. A loop whose ancestors provably always produce enough tasks
// can never run more than one worker, so its
// `select(w > 1, parallel, serial)` is folded to `serial`. If such a loop
// also contains a single function and its step is not required (no alignment
// constraint), the loop is pure overhead: one kernel call per tile with
// nothing to interleave or parallelize. Setting its step to the full extent
// makes it a provable single iteration, which slinky then folds away,
// leaving one kernel call over the whole range.
void compute_workers(ynn::slinky_globals& globals, int max_threads,
                     std::vector<loop_level>& global_loop_nest,
                     const std::vector<int>& funcs_in_level) {
  ynn::schedule_tuner* tuner = ynn::schedule_tuner::get();

  // Enough tasks to have good load balancing.
  slinky::index_t target_task_count = max_threads > 1 ? max_threads * 2 : 1;
  if (tuner && max_threads > 1) {
    // 0: 2x threads (the default), 1: 1x, 2: 4x, 3: 8x.
    static constexpr int multipliers[] = {2, 1, 4, 8};
    target_task_count =
        max_threads *
        multipliers[tuner->choose("task_target", "task_target", 4, 0)];
  }

  // A guaranteed lower bound of the number of iterations of loop level `l`:
  // ceil_div(lower bound of extent, upper bound of step), or 1 when either
  // bound is unknown (a scheduled loop runs at least one iteration).
  auto min_iterations = [&](const loop_level& l) -> slinky::index_t {
    std::optional<slinky::index_t> extent_lb =
        slinky::evaluate_constant_lower_bound(l.extent);
    std::optional<slinky::index_t> step_ub =
        slinky::evaluate_constant_upper_bound(resolve_let_var(globals, l.step));
    if (extent_lb && step_ub && *step_ub > 0) {
      return std::max<slinky::index_t>(slinky::ceil_div(*extent_lb, *step_ub),
                                       1);
    }
    return 1;
  };

  // For each loop, whether any of its descendants can be parallelized: a pure
  // or partial-reduction ("r") loop that is not provably a single iteration.
  // Serial reduction ("k") loops never count, they run inside one task by
  // construction. Loops are appended after their parent, so a single backward
  // pass propagates each loop's answer up to its parent.
  std::vector<bool> parallel_in_subtree(global_loop_nest.size(), false);
  for (int i = global_loop_nest.size() - 1; i >= 0; --i) {
    const loop_level& l = global_loop_nest[i];
    const slinky::expr once = is_single_iteration(globals, l);
    const bool is_parallel =
        !globals.is_reduction_dim(l.loop_id.var) && once.defined() &&
        !slinky::prove_true(once, globals.fact_bounds, globals.fact_alignment);
    if (l.parent >= 0 && (is_parallel || parallel_in_subtree[i])) {
      parallel_in_subtree[l.parent] = true;
    }
  }

  // The number of tasks the loops from the root down to (and including) each
  // loop can produce. Serial loops (reductions) run their iterations within
  // one task, so they don't contribute to this count. `tasks_lb` is a
  // guaranteed constant lower bound of the same quantity.
  std::vector<slinky::expr> tasks(global_loop_nest.size());
  std::vector<slinky::index_t> tasks_lb(global_loop_nest.size());
  for (size_t i = 0; i < global_loop_nest.size(); ++i) {
    loop_level& l = global_loop_nest[i];
    assert(l.parent < static_cast<int>(i));
    slinky::expr tasks_above = l.parent >= 0 ? tasks[l.parent] : 1;
    const slinky::index_t tasks_above_lb =
        l.parent >= 0 ? tasks_lb[l.parent] : 1;
    // Take the finer step a fused function proposed for this loop, if this
    // loop is where the nest has to get its parallelism. It isn't if the
    // ancestors already produce enough tasks, or if the subtree below has a
    // level that can be parallelized instead - splitting here would then cost
    // parallelism rather than add it, because the extra outer tasks push the
    // inner levels past the task target and turn them serial. The inner
    // levels are also the better place to split: finer tasks over contiguous
    // memory.
    if (l.proposed_step.defined() && max_threads > 1) {
      bool adopt =
          !parallel_in_subtree[i] && tasks_above_lb < target_task_count;
      if (tuner) {
        // 0: the heuristic above, 1: force adoption, 2: force rejection.
        switch (tuner->choose(tuner_func_name(globals, l.loop_id.func) + "." +
                                  globals.symbols.name(l.loop_id.var) +
                                  ".adopt",
                              "adopt", 3, 0)) {
          case 1: adopt = true; break;
          case 2: adopt = false; break;
        }
      }
      if (adopt) {
        // Behind a global so per-task closures reference a variable evaluated
        // once per invoke, not the whole select tree.
        l.step = globals.get(
            slinky::simplify(
                align_step(l.proposed_step, l.step_alignment),
                globals.fact_bounds, globals.fact_alignment),
            "s");
      }
    }
    // A loop that provably runs exactly one iteration is identical for any
    // number of functions inside it (required steps are excluded). Replacing
    // its step with the extent expression lets slinky prove the single
    // iteration and fold the loop away entirely.
    const bool elide_allowed =
        !l.step_is_required && globals.is_pure_dim(l.loop_id.var);
    // Serial reduction ("k") dims additionally qualify for the
    // single-iteration elision below (but not for the widening elisions):
    // with provably one iteration there is no accumulation blocking to
    // preserve.
    const bool single_iteration_elide_allowed =
        elide_allowed ||
        (!l.step_is_required && globals.is_reduction_dim(l.loop_id.var));
    const slinky::expr once = single_iteration_elide_allowed
                                  ? is_single_iteration(globals, l)
                                  : slinky::expr();
    if (once.defined() &&
        slinky::prove_true(once, globals.fact_bounds, globals.fact_alignment)) {
      l.step = slinky::max(slinky::simplify(l.extent), 1);
      l.workers = slinky::loop::serial;
      tasks[i] = tasks_above;
      tasks_lb[i] = tasks_above_lb;
    } else if (max_threads == 1 || globals.is_reduction_dim(l.loop_id.var)) {
      l.workers = slinky::loop::serial;
      // Reduction loops are left alone even when they contain a single
      // function: their step controls accumulation blocking, not just task
      // granularity.
      if (elide_allowed && i < funcs_in_level.size() &&
          funcs_in_level[i] <= 1) {
        l.step = slinky::max(l.extent, 1);
      }
      tasks[i] = tasks_above;
      tasks_lb[i] = tasks_above_lb;
    } else {
      // The loop is provably serial iff the loops above it always produce
      // enough tasks: w = ceil_div(target, tasks_above) <= 1 iff
      // tasks_above >= target.
      if (tasks_above_lb >= target_task_count) {
        l.workers = slinky::loop::serial;
        if (elide_allowed && i < funcs_in_level.size() &&
            funcs_in_level[i] <= 1) {
          l.step = slinky::max(l.extent, 1);
        }
      } else {
        slinky::expr w = globals.get(
            slinky::ceil_div(slinky::expr(target_task_count), tasks_above),
            "w");
        l.workers = slinky::simplify(slinky::select::make(
            w > 1, slinky::loop::parallel, slinky::loop::serial));
      }
      tasks[i] =
          slinky::simplify(tasks_above * slinky::ceil_div(l.extent, l.step));
      tasks_lb[i] = tasks_above_lb * min_iterations(l);
    }
  }
}

using source_region_map = std::map<std::pair<slinky::var, int>, int>;

int get_source_region(const source_region_map& source_regions, slinky::var buf,
                      int dim) {
  auto it = source_regions.find(std::make_pair(buf, dim));
  return it != source_regions.end() ? it->second : -1;
}

// Sharing a loop between the function that created it and a function being
// fused into it is two decisions, made in order by the two functions below:
// may this function's split cover the loop at all (find_matching_split), and
// what step does the shared loop end up with (reconcile_step).

// Find the split of `f` that covers `consumer_source_region`, or -1 if none
// does. `split_matched` marks the splits already used by outer loops.
//
// The splits don't have to be matched in their declared order: loops over pure
// dims carry no state across iterations, so they can be freely reordered, and
// splits which are not matched simply remain the function's own inner loops.
// Non-pure (reduction) splits do carry state across iterations (they
// accumulate into the same output), so they act as a fence: nothing is matched
// at or beyond the first one.
int find_matching_split(ynn::slinky_globals& globals, const slinky::func& f,
                        const std::vector<ynn::scheduling_split>& loop_splits,
                        const std::vector<bool>& split_matched,
                        int consumer_source_region,
                        const source_region_map& source_regions) {
  if (consumer_source_region == -1) return -1;

  // Whether the search has passed over an unmatched (and non-trivial) split,
  // i.e. matching a later split would reorder the function's loops.
  bool out_of_order = false;

  // Whether matching `split_i` here preserves this function's required
  // blocking order, i.e. no required split before it is still unmatched.
  auto keeps_required_order = [&](int split_i) {
    for (int prev = 0; prev < split_i; ++prev) {
      if (!split_matched[prev] && loop_splits[prev].step_is_required) {
        return false;
      }
    }
    return true;
  };
  // Whether the product of this function's reduction extents is provably below
  // a threshold. The value works for the benchmarks we have; it may well need
  // tuning, or replacing by something derived from the shapes.
  constexpr slinky::index_t small_reduction_threshold = 256;
  auto has_small_reduction = [&]() {
    slinky::index_t k_product = 1;
    for (const auto& out : f.outputs()) {
      for (int d = 0; d < static_cast<int>(out.dims.size()); ++d) {
        if (globals.is_pure_dim(out.dims[d])) continue;
        auto c = slinky::as_constant(
            slinky::simplify(out.buffer->dim(d).bounds.extent()));
        if (!c) return false;
        k_product *= *c;
      }
    }
    return k_product <= small_reduction_threshold;
  };

  for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
    if (split_matched[split_i]) continue;
    const ynn::scheduling_split& split = loop_splits[split_i];
    if (!globals.is_pure_dim(split.var)) {
      // We don't want to fuse a reduction dimension because it is likely being
      // broadcasted here, and we don't reorder other splits across it either.
      break;
    }
    if (split.step_is_required && out_of_order &&
        !(keeps_required_order(split_i) || has_small_reduction())) {
      // Matching a required split out of order is fine for any step, but it
      // must not reorder the function's required loops relative to each
      // other: that blocking was chosen deliberately, and inverting it makes
      // the function re-read its inputs. The exception is a function whose
      // reduction is small enough that re-reading costs nothing.
      continue;
    }
    // Map the producer's loop variable back to its output dimension index.
    auto [producer_buf, producer_dim] = find_output_dim(&f, split.var);

    // Instead of comparing forward extents (which causes false positives for
    // unrelated constant extents), we check if both loops share the exact same
    // inferred source region identifier.
    if (producer_dim != -1 && producer_buf.defined() &&
        get_source_region(source_regions, producer_buf, producer_dim) ==
            consumer_source_region) {
      return split_i;
    }
    out_of_order = true;
  }
  return -1;
}

// Decide the step of a loop now shared by its owner and `split`. Each side
// makes a claim on the step, and the stronger claim wins:
//
//   required + required : reconcile with the lcm, so the loop is an integer
//                         number of *both* tiles (two producers can require
//                         different tiles for a shared loop, e.g. the two
//                         attention matmuls pick different query tiles).
//   required + anything : the required step, which is a kernel's blocking.
//   chosen   + no claim : propose the chosen step, see below.
//   otherwise           : keep the loop's step. When both sides computed a
//                         real split, each is only meaningful within its own
//                         cache-budget allocation, so combining them (e.g. by
//                         taking the min) degenerates to tiny steps.
//
// "No claim" means the loop runs a single full-extent iteration, see
// is_single_iteration(). `loop_splits` are all of the matching function's
// splits, used to size its iteration space.
void reconcile_step(ynn::slinky_globals& globals, loop_level& loop,
                    const ynn::scheduling_split& split,
                    const std::vector<ynn::scheduling_split>& loop_splits) {
  // A function whose kernel needs aligned crops (see step_alignment) imposes
  // that alignment on the shared loop: whatever step the reconciliation below
  // ends up with, it must be a multiple of the alignment. Multiplying an
  // unaligned step by the alignment keeps it an integer number of the
  // original tiles (like the lcm rule for two required steps), so it stays
  // valid for every function already in the loop. The alignment is also
  // remembered on the loop so steps installed later (a proposed step adopted
  // by compute_workers) respect it too.
  loop.step_alignment = std::max(loop.step_alignment, split.step_alignment);
  auto aligned = [&](slinky::expr step) {
    const slinky::index_t a = loop.step_alignment;
    if (a > 1 && step.defined()) {
      step = slinky::simplify(slinky::select(step % a == 0, step, step * a),
                              globals.fact_bounds, globals.fact_alignment);
    }
    return step;
  };
  // Align the loop's existing step up front, so every reconciliation outcome
  // below (including keeping the step as is) leaves the loop aligned. A
  // pr_split-variable step is left alone: rewriting it here would desync the
  // partial reduction bounds pinned to that variable, and a partial
  // reduction's loop cannot match an alignment-carrying split anyway.
  if (loop.step_alignment > 1) {
    std::optional<slinky::var> v = slinky::as_variable(loop.step);
    if (!(v && globals.symbols.name(*v).rfind("pr_split", 0) == 0)) {
      loop.step = aligned(loop.step);
    }
  }
  if (split.step_is_required) {
    if (loop.step_pinned) {
      // Keep the pinned constant tile; see loop_level::step_pinned.
      loop.step_is_required = true;
      loop.proposed_step = slinky::expr();
      return;
    }
    if (loop.step_is_required &&
        !prove_true(split.step == loop.step, globals.fact_bounds,
                    globals.fact_alignment)) {
      // If the LCM overflows, it clamps at max index_t (assuming no
      // splitting).
      loop.step = aligned(lcm_sat(globals, loop.step, split.step));
    } else {
      const slinky::expr new_step = aligned(split.step);
      if (std::optional<slinky::var> v = slinky::as_variable(loop.step)) {
        // This is a special variable which defines partial reduction bounds,
        // so we need to override to match the loop step.
        if (globals.symbols.name(*v).rfind("pr_split", 0) == 0) {
          globals.update_let(*v, new_step);
        }
      }
      loop.step = new_step;
    }
    loop.step_is_required = true;
    // A required step is a kernel's blocking, so it outranks any step an
    // earlier function proposed for this loop.
    loop.proposed_step = slinky::expr();
    return;
  }
  if (loop.step_is_required || !loop.step.defined() || !split.step.defined()) {
    return;
  }
  // Only a pure dim's step is free to change. A partial reduction's "r" loop
  // is steppable, but its step is coupled to the reduction buffer's
  // fold_factor (the kernel's accumulation chunk), so a different step would
  // desync the accumulation and corrupt results.
  if (!globals.is_pure_dim(split.var) ||
      !globals.is_pure_dim(loop.loop_id.var)) {
    return;
  }
  // This function computed a real split for a loop its owner left as a single
  // task. Propose the split, so that a producer fused into the loop keeps its
  // parallelism instead of running serially - without this, a norm's reduce
  // stages serialize the whole chain they fuse into.
  //
  // Several functions can match the same loop, and the last proposal wins. A
  // function whose split is already the loop's step is asking for nothing, so
  // drop it rather than let it displace an earlier real proposal.
  if (prove_true(split.step == loop.step, globals.fact_bounds,
                 globals.fact_alignment)) {
    return;
  }
  // The function's splits span its whole iteration space, so their extent
  // product is the work that would be divided up. Below this threshold the
  // extra task dispatches cost more than the split saves; the value was found
  // experimentally.
  constexpr slinky::index_t min_work_per_split = 128 * 1024;
  slinky::expr work = 1;
  for (const ynn::scheduling_split& ls : loop_splits) {
    work = work * ls.extent;
  }
  // Both conditions can depend on the runtime shape, so they go into the
  // proposal as a select, which folds away for static shapes.
  loop.proposed_step = slinky::select(
      is_single_iteration(globals, loop) && min_work_per_split <= work,
      split.step, loop.step);
}

}  // namespace

// Logically this function has multiple separate blocks:
// 1) infer symbolic source regions for all buffers to ensure loops are only
//    fused if they share a common source region origin.
// 2) computing a list of possible compute_at locations for a given function.
//    This is a very concrete thing and doesn't require any heuristics.
// 3) using the set of locations from 2) decide if we want for this function
//    to be computed at root or at one of the existing loops based on the
//    available information such as scheduling_info attached to the function
//    or source regions inferred in 1).
// 4) if we decide to share the loop location possibly update loop parameters
//    such as step based on the specific of the given function (this is pretty
//    much a no-op right now and is solely defined by a "parent" function of
//    the loop, but we can use it in the future to figure out, for example, a
//    step size based on the *all* functions which were assigned to the loop).
// 5) potentially add new loop(s) into the loop nest based on a given
//    function.
// 6) based on compute locations computed in 2) - 5), set up the
//    func-s. This is done in a separate loop once all of the functions from
//    the pipeline were processed.
void ynn_runtime::schedule() {
  // When the autotuner is active, identify this pipeline by a hash of its
  // structure (buffer symbols and split counts), so recorded decisions can be
  // replayed against rebuilds of the same pipeline.
  ynn::schedule_tuner* tuner = ynn::schedule_tuner::get();
  if (tuner) {
    uint64_t key = 0xcbf29ce484222325ULL;
    auto mix = [&key](uint64_t v) { key = (key ^ v) * 0x100000001b3ULL; };
    for (const slinky::func& f : funcs) {
      for (const auto& out : f.outputs()) {
        for (char c : globals.symbols.name(out.sym())) {
          mix(static_cast<uint8_t>(c));
        }
      }
      const auto* sched =
          static_cast<const ynn::scheduling_info*>(f.user_data());
      mix(sched ? sched->loop_splits.size() + 1 : 0);
    }
    tuner->begin_pipeline(key);
  }

  // `thread_count()` reports the number of background worker threads. The
  // thread that invokes the runtime also participates as a worker (it runs
  // tasks while waiting in `thread_pool::wait_for`), so the effective
  // parallelism is one more than the reported count. Without this `+ 1`, a
  // pool with a single background thread (two threads of execution in total)
  // would be scheduled serially, and every other size would be sized one
  // worker short.
  const int max_threads = threadpool() ? threadpool()->thread_count() + 1 : 1;

  // This a list of indices of consumers of a given buffer.
  std::map<slinky::var, std::vector<int>> consumers;
  // This is a tree representing a global loop nest of a whole pipeline so
  // far. For efficiency and convenience, it's stored as an array of nodes
  // with auxiliary structures using indices to refer to the loop levels.
  std::vector<loop_level> global_loop_nest;

  std::vector<scheduling_data> func_scheduling_data(funcs.size());

  // Maps {buffer_sym, dim_index} to its inferred source region unique
  // identifier.
  source_region_map source_regions = infer_source_regions(funcs);

  // Slinky doesn't allocate the pipeline's output buffers, and as a result it
  // also never crops them to the region a loop iteration needs. Fusing the
  // producer of an external output into a loop of one of its consumers would
  // therefore run the producer (and everything it depends on) over the
  // whole output on *every* iteration of that loop.
  std::set<slinky::var> external_output_syms;
  for (const ynn_runtime_value& value : values) {
    if (value.is_valid() && value.is_external_output()) {
      external_output_syms.insert(value.symbol);
    }
  }

  for (int i = funcs.size() - 1; i >= 0; --i) {
    slinky::func& f = funcs[i];
    scheduling_data& sched_data = func_scheduling_data[i];
    std::vector<int>& loop_nest = sched_data.loop_nest;
    // First of all, we need to find where this function can be scheduled
    // based on its customers. The options are a range of loop levels starting
    // from outermost root to a common subnest of its consumer loop nests.
    // In order to find a common subnest, we iterate over loop nests of the
    // customers and find a common prefix. This also can be viewed as finding
    // a least common ancestor.
    bool loop_nest_initialized = false;
    for (const auto& output : f.outputs()) {
      // Find common subnest of the consumers.
      for (int consumer_index = 0;
           consumer_index < consumers[output.buffer->sym()].size();
           consumer_index++) {
        int consumer = consumers[output.buffer->sym()][consumer_index];
        const std::vector<int>& consumer_loop_nest =
            func_scheduling_data[consumer].loop_nest;
        if (!loop_nest_initialized) {
          loop_nest = consumer_loop_nest;
          loop_nest_initialized = true;
          continue;
        }
        if (consumer_loop_nest.size() < loop_nest.size()) {
          loop_nest.erase(loop_nest.begin() + consumer_loop_nest.size(),
                          loop_nest.end());
        }
        for (int j = 0;
             j < std::min(consumer_loop_nest.size(), loop_nest.size()); ++j) {
          if (loop_nest[j] != consumer_loop_nest[j]) {
            loop_nest.erase(loop_nest.begin() + j, loop_nest.end());
            break;
          }
        }
      }
    }

    int compute_at = -1;
    ynn::scheduling_info* sched =
        static_cast<ynn::scheduling_info*>(f.user_data());

    // If they are not matching and the extents of f are smaller than parent
    // we are risking over-compute. If it doesn't have schedule info then we
    // just compute at the innermost location.
    if (sched && !sched->loop_splits.empty()) {
      std::vector<ynn::scheduling_split>& loop_splits = sched->loop_splits;

      // Autotuner decision: scale this function's split steps. Only pure
      // dims: a reduction ("k") loop's step controls accumulation blocking,
      // and a partial reduction ("r") loop's step is coupled to the reduction
      // buffer's fold_factor. A required step is a kernel blocking, so only
      // integer multiples keep the alignment guarantees derived from it;
      // other steps are cache splits that can scale both ways.
      if (tuner) {
        const std::string fkey = tuner_func_name(globals, &f);
        for (ynn::scheduling_split& split : loop_splits) {
          if (!globals.is_pure_dim(split.var) || !split.step.defined()) {
            continue;
          }
          const std::string key =
              fkey + "." + globals.symbols.name(split.var) + ".scale";
          if (split.step_is_required) {
            // 0: x1, 1: x2, 2: x4.
            static constexpr int multipliers[] = {1, 2, 4};
            int m = multipliers[tuner->choose(key, "req_scale", 3, 0)];
            if (m != 1) split.step = split.step * m;
          } else {
            // 0: x1, 1: x2, 2: x4, 3: /2, 4: /4.
            switch (tuner->choose(key, "scale", 5, 0)) {
              case 1: split.step = split.step * 2; break;
              case 2: split.step = split.step * 4; break;
              case 3: split.step = slinky::max(split.step / 2, 1); break;
              case 4: split.step = slinky::max(split.step / 4, 1); break;
            }
            split.step = align_step(split.step, split.step_alignment);
          }
        }
      }

      // Make sure that extents of the dims belonging to subnest match.
      // Reverse to simplify indexing below.
      std::reverse(loop_splits.begin(), loop_splits.end());

      std::vector<bool>& split_matched = sched_data.split_matched;
      split_matched.assign(loop_splits.size(), false);

      // Splits with a provable extent of 1 don't need a loop of their own and
      // must not become levels of the global loop nest: a degenerate level
      // would block the functions scheduled later from matching the loops
      // behind it. Treat them as trivially matched, so they are neither
      // considered for matching nor appended to the nest.
      for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
        if (prove_true(loop_splits[split_i].extent == 1, globals.fact_bounds,
                       globals.fact_alignment)) {
          split_matched[split_i] = true;
        }
      }

      // Walk the loop nest from the outermost loop inwards, sharing each loop
      // with a split of this function that covers the same source region (see
      // find_matching_split and reconcile_step). We must stop at the first
      // loop of the nest we can't cover: computing the function inside a loop
      // which doesn't slice its output would recompute the function on every
      // iteration of that loop.
      //
      // Un-fuse rule: fusing this function into a consumer nest that cannot
      // parallelize serializes all of its work. The canonical case is a
      // blockwise-quantized dot chain rooted at a cross-block reduction: the
      // reduction loop is serial by construction and the consumer's pure
      // loops are single cache-sized tiles, so the dot -- the bulk of the
      // work -- runs on one thread. When this function's own loops provably
      // supply the parallelism the nest lacks, and materializing its output
      // is affordable, computing it at root wins: measured 2.4-3x on
      // blockwise int4/int8 dot chains for medium batch sizes and neutral
      // elsewhere. Deliberately conservative: it acts only on parallelism
      // this function's *current* splits prove (per iterations_lower_bound),
      // so cases where the win requires re-splitting the function finer
      // (very small batches, where the searched optimum splits the block
      // dimension below the cache-derived split) are left to the default
      // schedule for now.
      int rule_max_compute_at = static_cast<int>(loop_nest.size());
      if (getenv("YNN_UNFUSE_DEBUG") && !loop_nest.empty()) {
        std::optional<slinky::index_t> bytes =
            output_bytes_upper_bound(f, loop_splits);
        std::string splits_desc;
        for (const ynn::scheduling_split& s : loop_splits) {
          std::optional<slinky::index_t> ub =
              slinky::evaluate_constant_upper_bound(s.extent);
          splits_desc += " " + globals.symbols.name(s.var) +
                         (s.step_is_required ? "!" : "") + "=" +
                         (ub ? std::to_string(*ub) : std::string("sym"));
        }
        fprintf(stderr,
                "unfuse? %s nest_tasks=%lld own_tasks=%lld bytes=%lld "
                "threads=%d nest=%zu splits:%s\n",
                tuner_func_name(globals, &f).c_str(),
                (long long)nest_tasks_lower_bound(globals, global_loop_nest,
                                                  loop_nest),
                (long long)split_tasks_lower_bound(globals, loop_splits),
                bytes ? (long long)*bytes : -1, max_threads, loop_nest.size(),
                splits_desc.c_str());
      }
      // A function with a required (kernel-blocking) split is excluded: an
      // out-of-order required match retiles the consumer's loops with the
      // kernel's own blocking, so fusing it can parallelize the nest that
      // looks serial before the match (e.g. a dot fusing into its rescale
      // chain).
      const bool has_required_split =
          std::any_of(loop_splits.begin(), loop_splits.end(),
                      [](const ynn::scheduling_split& s) {
                        return s.step_is_required;
                      });
      // The nest must be provably starved (fewer tasks than threads), and
      // this function's own parallelism must decisively beat it -- a modest
      // advantage (e.g. a few-block bs256 chain with 2-8 tasks either way)
      // does not pay for the materialization.
      const slinky::index_t nest_tasks =
          nest_tasks_lower_bound(globals, global_loop_nest, loop_nest);
      const slinky::index_t own_tasks =
          split_tasks_lower_bound(globals, loop_splits);
      // The materialized intermediate is written and then re-read by the
      // consumer; that traffic only streams efficiently if the contiguous
      // (innermost output) dimension gives reasonably long runs. With tiny
      // runs the re-read is strided and the materialization costs more than
      // the parallelism recovers (measured 0.8-0.9x on n=32 x 256-block
      // chains whose n=128+ twins win 2x).
      const slinky::index_t contiguous_run_bytes = [&]() -> slinky::index_t {
        if (f.outputs().empty()) return 0;
        const auto& out = f.outputs()[0];
        std::optional<slinky::index_t> elem =
            slinky::as_constant(out.buffer->elem_size());
        if (!elem || out.dims.empty()) return 0;
        for (const ynn::scheduling_split& s : loop_splits) {
          if (s.var == out.dims[0]) {
            std::optional<slinky::index_t> ub =
                slinky::evaluate_constant_upper_bound(s.extent);
            return ub ? *ub * *elem : 0;
          }
        }
        return 0;
      }();
      if (unfuse_rule_enabled() && max_threads > 1 && !loop_nest.empty() &&
          !has_required_split && nest_tasks < max_threads &&
          own_tasks >= 2 * max_threads &&
          own_tasks >= 16 * nest_tasks &&
          contiguous_run_bytes >= 512) {
        std::optional<slinky::index_t> bytes =
            output_bytes_upper_bound(f, loop_splits);
        if (bytes && *bytes <= unfuse_bytes_budget()) {
          rule_max_compute_at = 0;
          // The consumer nest keeps the remaining (reduction) phase; its
          // single-tile pure loops get a finer proposed step so that phase
          // can parallelize too. compute_workers adopts a proposal only if
          // the nest still needs the tasks.
          for (int i : loop_nest) {
            loop_level& l = global_loop_nest[i];
            if (!globals.is_pure_dim(l.loop_id.var) || l.step_is_required ||
                l.proposed_step.defined() || !l.step.defined()) {
              continue;
            }
            l.proposed_step =
                align_step(slinky::max(l.step / 4, 1), l.step_alignment);
          }
        }
      }

      // Symbolic-batch arm of the un-fuse rule. When exactly one pure
      // dimension (the batch) has no constant bound but the rest of the
      // iteration space is constant -- the shape of an inference graph with
      // static weights and a dynamic batch -- the arm above cannot prove its
      // task counts or byte bound. A batch-invariant version of the same
      // schedule exists: fuse this function only into the consumer loop that
      // slices the batch dimension, force that loop to a constant tile, and
      // give the function's constant-extent splits real steps. The
      // materialized intermediate is then one batch tile (constant bytes for
      // any runtime batch); at small batch the inner (e.g. block) loops
      // parallelize and at large batch the tile loop does.
      if (unfuse_rule_enabled() && max_threads > 1 && !loop_nest.empty() &&
          !has_required_split && rule_max_compute_at != 0 &&
          nest_tasks < max_threads && contiguous_run_bytes >= 512) {
        const std::optional<slinky::index_t> elem =
            f.outputs().empty()
                ? std::nullopt
                : slinky::as_constant(f.outputs()[0].buffer->elem_size());
        std::vector<bool> is_symbolic(loop_splits.size(), false);
        int num_symbolic = 0;
        slinky::index_t const_potential = 1;
        slinky::index_t row_bytes = elem.value_or(0);
        for (int i = 0; i < static_cast<int>(loop_splits.size()); ++i) {
          const ynn::scheduling_split& s = loop_splits[i];
          if (!globals.is_pure_dim(s.var)) continue;
          std::optional<slinky::index_t> ub =
              slinky::evaluate_constant_upper_bound(s.extent);
          if (!ub) {
            is_symbolic[i] = true;
            ++num_symbolic;
          } else {
            const_potential = slinky::mul_sat(const_potential, *ub);
            row_bytes =
                slinky::mul_sat(row_bytes, std::max<slinky::index_t>(*ub, 1));
          }
        }
        const slinky::index_t tile = unfuse_batch_tile();
        if (elem && num_symbolic > 0 && const_potential >= 2 * max_threads &&
            slinky::mul_sat(row_bytes, tile) <= unfuse_bytes_budget()) {
          // Walk the consumer loops the same way the fusion walk below will,
          // continuing only while they match this function's symbolic splits.
          // Every symbolic split must be matched within that prefix: an
          // unmatched one would leave a full symbolic extent in the
          // materialized tile.
          std::vector<bool> matched = split_matched;
          std::vector<int> prefix;  // loop_nest indices, outermost first
          int last_symbolic_loop = -1;
          bool serialized_parallelism = false;
          for (size_t i = 0; i < loop_nest.size(); ++i) {
            loop_level& l = global_loop_nest[loop_nest[i]];
            const bool is_pr = [&]() {
              std::optional<slinky::var> v = slinky::as_variable(l.step);
              return v &&
                     globals.symbols.name(*v).rfind("pr_split", 0) == 0;
            }();
            if (l.step_is_required || is_pr) break;
            auto [consumer_buf, consumer_dim] =
                find_output_dim(l.loop_id.func, l.loop_id.var);
            const int region =
                consumer_dim != -1 && consumer_buf.defined()
                    ? get_source_region(source_regions, consumer_buf,
                                        consumer_dim)
                    : -1;
            const int m = find_matching_split(globals, f, loop_splits, matched,
                                              region, source_regions);
            if (m == -1) break;
            if (is_symbolic[m]) {
              if (num_symbolic == 0) break;
              matched[m] = true;
              prefix.push_back(static_cast<int>(i));
              last_symbolic_loop = static_cast<int>(i);
              --num_symbolic;
              continue;
            }
            // Past the symbolic dims: full fusion would match this constant
            // split into the consumer's loop. If that loop is a serial
            // reduction and the split is wide, fusion serializes exactly the
            // parallelism un-fusing recovers -- the signal this arm needs.
            if (num_symbolic == 0 &&
                globals.is_reduction_dim(l.loop_id.var)) {
              std::optional<slinky::index_t> ub =
                  slinky::evaluate_constant_upper_bound(loop_splits[m].extent);
              if (ub && *ub >= 2 * max_threads) {
                serialized_parallelism = true;
              }
              break;
            }
            matched[m] = true;
          }
          if (getenv("YNN_UNFUSE_DEBUG")) {
            fprintf(stderr,
                    "unfuse-sym? %s sym_left=%d prefix=%zu serialized=%d "
                    "potential=%lld row_bytes=%lld run=%lld\n",
                    tuner_func_name(globals, &f).c_str(), num_symbolic,
                    prefix.size(), (int)serialized_parallelism,
                    (long long)const_potential, (long long)row_bytes,
                    (long long)contiguous_run_bytes);
          }
          if (num_symbolic == 0 && !prefix.empty() && serialized_parallelism) {
            rule_max_compute_at = static_cast<int>(prefix.size());
            // Constant tiles on the fused symbolic loops: the innermost gets
            // the batch tile, outer ones step by 1, so the materialized tile
            // is row_bytes * tile for any number of symbolic dimensions.
            for (int i : prefix) {
              loop_level& l = global_loop_nest[loop_nest[i]];
              const slinky::index_t step = i == last_symbolic_loop ? tile : 1;
              l.step = align_step(slinky::expr(step), l.step_alignment);
              l.proposed_step = slinky::expr();
              l.step_is_required = true;
              l.step_pinned = true;
            }
            // Realize this function's own parallelism: constant-extent pure
            // splits that made no real splitting decision get a constant
            // step sized for the thread pool.
            for (ynn::scheduling_split& s : loop_splits) {
              if (!globals.is_pure_dim(s.var)) continue;
              std::optional<slinky::index_t> ub =
                  slinky::evaluate_constant_upper_bound(s.extent);
              if (!ub || *ub < 2) continue;
              if (iterations_lower_bound(globals, s.extent, s.step) > 1) {
                continue;
              }
              const slinky::index_t step = std::max<slinky::index_t>(
                  slinky::ceil_div<slinky::index_t>(*ub, 4 * max_threads), 1);
              s.step = align_step(slinky::expr(step), s.step_alignment);
            }
            // Tile the consumer's remaining single-tile pure loops for the
            // reduction phase, as in the arm above.
            for (size_t i = prefix.size(); i < loop_nest.size(); ++i) {
              loop_level& l = global_loop_nest[loop_nest[i]];
              if (!globals.is_pure_dim(l.loop_id.var) || l.step_is_required ||
                  l.proposed_step.defined() || !l.step.defined()) {
                continue;
              }
              l.proposed_step =
                  align_step(slinky::max(l.step / 4, 1), l.step_alignment);
            }
          }
        }
      }

      // Autotuner decision: cap how deep this function fuses. The default is
      // the un-fuse rule's answer (usually as deep as the splits can match);
      // 0 means compute root.
      int max_compute_at = rule_max_compute_at;
      if (tuner && !loop_nest.empty()) {
        max_compute_at =
            tuner->choose(tuner_func_name(globals, &f) + ".fuse", "fuse",
                          static_cast<int>(loop_nest.size()) + 1,
                          rule_max_compute_at);
      }
      compute_at = 0;
      while (compute_at < max_compute_at) {
        loop_level& global_loop = global_loop_nest[loop_nest[compute_at]];
        // Map the consumer's loop variable back to its output dimension
        // index.
        auto [consumer_buf, consumer_dim] =
            find_output_dim(global_loop.loop_id.func, global_loop.loop_id.var);
        const int consumer_source_region =
            consumer_dim != -1 && consumer_buf.defined()
                ? get_source_region(source_regions, consumer_buf, consumer_dim)
                : -1;

        const int matched_split =
            find_matching_split(globals, f, loop_splits, split_matched,
                                consumer_source_region, source_regions);
        if (matched_split == -1) {
          break;
        }
        split_matched[matched_split] = true;
        reconcile_step(globals, global_loop, loop_splits[matched_split],
                       loop_splits);
        compute_at++;
      }
      // Remove the inner part of the loop nest which we were not able to
      // match.
      loop_nest.erase(loop_nest.begin() + compute_at, loop_nest.end());
    }

    // A function producing an external output cannot be fused into a loop of
    // its consumers without recomputing all of it on every iteration, see
    // `external_output_syms` above.
    const bool produces_external_output =
        !loop_nest.empty() &&
        std::any_of(f.outputs().begin(), f.outputs().end(),
                    [&](const slinky::func::output& o) {
                      return external_output_syms.count(o.buffer->sym()) > 0;
                    });

    if ((sched && sched->force_root) || produces_external_output) {
      compute_at = 0;
      if (sched) {
        sched_data.split_matched.assign(sched->loop_splits.size(), false);
      }
      loop_nest.clear();
    }

    sched_data.compute_at = compute_at;

    // NOTE: potentially we could also track how much specific loop are
    // computing by keeping a sum of compute amounts for each of the functions
    // inside of this loop and only schedule loops which have more
    // computations than certain threshold.
    if (sched && !sched->loop_splits.empty()) {
      const std::vector<ynn::scheduling_split>& loop_splits =
          sched->loop_splits;
      // Update the global loop nest by adding the unmatched loops of this
      // function, preserving their relative order.
      for (int j = 0; j < loop_splits.size(); j++) {
        if (sched_data.split_matched[j]) continue;
        const ynn::scheduling_split& dim = loop_splits[j];
        const int parent = loop_nest.empty() ? -1 : loop_nest.back();
        global_loop_nest.push_back({{&f, dim.var},
                                    dim.extent,
                                    dim.step,
                                    dim.step_is_required,
                                    dim.step_alignment,
                                    /*step_pinned=*/false,
                                    parent});
        loop_nest.push_back(global_loop_nest.size() - 1);
      }
    }

    // Record which buffers this function is consuming.
    for (const auto& input : f.inputs()) {
      consumers[input.buffer->sym()].push_back(i);
    }
  }

  // A function executes inside every loop of its (final) loop nest, so the
  // number of functions inside a loop level is the number of loop nests it
  // appears in. A count of 1 means the level only contains the function that
  // created it.
  std::vector<int> funcs_in_level(global_loop_nest.size(), 0);
  for (const scheduling_data& sched_data : func_scheduling_data) {
    for (int level : sched_data.loop_nest) {
      funcs_in_level[level]++;
    }
  }
  compute_workers(globals, max_threads, global_loop_nest, funcs_in_level);

  // Use previously computed information to actually schedule the functions.
  for (int i = funcs.size() - 1; i >= 0; --i) {
    slinky::func& f = funcs[i];
    const scheduling_data& sched_data = func_scheduling_data[i];
    ynn::scheduling_info* sched =
        static_cast<ynn::scheduling_info*>(f.user_data());
    int compute_at = sched_data.compute_at;
    // Now we know a compute_at location of this function
    if (compute_at == 0) {
      f.compute_root();
    } else {
      const std::vector<int>& loop_nest = sched_data.loop_nest;
      if (compute_at > 0) {
        const slinky::loop_id& lid =
            global_loop_nest[loop_nest[compute_at - 1]].loop_id;
        f.compute_at(lid);
      }
      if (!sched || sched->scheduled_buffers.empty()) {
        f.store_outputs_innermost();
      } else {
        for (auto& b : sched->scheduled_buffers) {
          if (b.store_at_min_depth == 0) {
            b.buffer->store_at({&funcs[i], slinky::var()});
          } else if (b.store_at_min_depth < loop_nest.size()) {
            const slinky::loop_id& lid =
                global_loop_nest[loop_nest[b.store_at_min_depth - 1]].loop_id;
            b.buffer->store_at(lid);
          } else {
            b.buffer->store_root();
          }
        }
      }
    }

    if (sched && !sched->loop_splits.empty()) {
      std::vector<ynn::scheduling_split>& loop_splits = sched->loop_splits;
      const std::vector<int>& loop_nest = sched_data.loop_nest;
      const std::vector<bool>& split_matched = sched_data.split_matched;
      // Reverse it back.
      std::reverse(loop_splits.begin(), loop_splits.end());

      // The loops of this function are its unmatched splits (matched splits
      // are loops of the function this one was fused into). The j-th
      // unmatched split from the innermost corresponds to the j-th innermost
      // entry of the loop nest, which tracks the (possibly updated by other
      // fused functions) step of the loop and the globally computed workers.
      std::vector<slinky::func::loop_info> loops;
      for (int i = 0; i < loop_splits.size(); ++i) {
        // loop_splits was reversed back to its original order, but
        // split_matched is indexed in the reversed order used for matching.
        if (split_matched[loop_splits.size() - 1 - i]) continue;
        const ynn::scheduling_split& dim = loop_splits[i];
        const loop_level& level =
            global_loop_nest[loop_nest[loop_nest.size() - loops.size() - 1]];
        loops.push_back({dim.var, level.step, level.workers});
      }

      f.loops(std::move(loops));
    }
  }

  if (tuner) {
    tuner->end_pipeline();
  }
}

slinky::buffer_expr_ptr ynn_runtime::null_buffer() {
  if (!null_buffer_) {
    slinky::var null(globals.symbols, "null");
    null_buffer_ = slinky::buffer_expr::make_constant(
        null, slinky::raw_buffer::make(0, 0));
  }
  return null_buffer_;
}

namespace {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// This generates a call that populates all the shapes of the xnn_values using
// the forward bounds expressions.
auto make_reshape_impl(ynn_runtime* runtime) {
  // Gather only what we need for capturing in the callback lambda.
  return [runtime](const slinky::call_stmt*,
                   slinky::eval_context& ctx) -> slinky::index_t {
    int errors = 0;
    for (const ynn_node& node : runtime->subgraph->nodes) {
      if (!node.is_valid()) continue;
      for (const auto& check : node.checks) {
        if (!slinky::evaluate(check.condition, ctx)) {
          std::stringstream error;
          for (const auto& i : check.message) {
            std::visit(overloaded{
                           [&](const char* s) { error << s; },
                           [&](slinky::expr e) { error << evaluate(e, ctx); },
                           [&](ynn_node::input_idx i) {
                             error << "input " << i.idx << " (id "
                                   << node.inputs[i.idx] << ")";
                           },
                           [&](ynn_node::output_idx i) {
                             error << "output " << i.idx << " (id "
                                   << node.outputs[i.idx] << ")";
                           },
                       },
                       i);
          }
          YNN_LOG_ERROR() << "Error in node '" << node.name() << ": "
                          << error.str();
          ++errors;
        }
      }
    }
    if (errors) {
      return ynn_status_invalid_parameter;
    }

    for (auto& i : runtime->values) {
      if (!i.is_valid()) continue;
      if (i.is_external_output()) {
        assert(i.data);
        assert(i.data->rank == i.rank());
        std::vector<slinky::expr> phys_extents = i.physical_extents();
        for (size_t d = 0; d < i.rank(); ++d) {
          slinky::expr extent_d = i.physical_extent(d);
          if (extent_d.defined()) {
            i.data->mutable_dim(d).set_min_extent(0, evaluate(extent_d, ctx));
          } else {
            i.data->mutable_dim(d).set_min_extent(0, 1);
          }
        }
        ynn::init_buffer_strides(*i.data);
      }
    }
    return 0;
  };
}

#ifdef YNN_ENABLE_PERFETTO
// TODO(dsharlet): We need a better way to control tracing output.
const char* get_trace_filename() { return getenv("YNN_TRACE"); }
#endif

#ifdef YNN_ENABLE_TSL_PROFILER
bool ynn_traceme_enabled() {
  // We can't use `TraceMe::Active` here, because it returns false when called,
  // even if it would later return true when we actually want to trace. We
  // should also gate this behind an extra flag because our tracing might be a
  // lot higher frequency than other xprof tracing.
  const char* traceme = getenv("YNN_TRACEME");
  return traceme && strcmp(traceme, "0") != 0;
}
#endif

// Slinky will automatically place allocates on the stack if the allocation is
// smaller than this threshold.
constexpr size_t auto_stack_threshold = 64 * 1024;

}  // namespace

extern "C" {

ynn_runtime::ynn_runtime(ynn::ref_count<const ynn_subgraph> subgraph,
                         slinky::thread_pool* threadpool, uint32_t flags)
    : subgraph(subgraph), flags(flags), globals(subgraph->globals) {
  // Implement our required alignment for heap allocations.
  eval_config.allocate = [](slinky::var sym, slinky::raw_buffer* buffer) {
    return buffer->allocate(YNN_ALLOCATION_ALIGNMENT);
  };
  eval_config.free = [](slinky::var sym, slinky::raw_buffer* buffer,
                        void* ptr) { std::free(ptr); };
  eval_config.thread_pool = threadpool;
  // Slinky's default check failure handler calls std::abort(), don't let that
  // happen here.
  eval_config.check_failed = [](const slinky::expr& e) {
    YNN_LOG_ERROR() << "Check failed";
  };
  eval_config.call_failed = [](const slinky::call_stmt* c) {
    YNN_LOG_ERROR() << c->attrs.name << " failed";
  };
  eval_config.base_alignment = YNN_ALLOCATION_ALIGNMENT;
  eval_config.auto_stack_threshold = auto_stack_threshold;

#ifdef YNN_ENABLE_PERFETTO
  if (ynn::perfetto_session::global()) {
    eval_config.trace_begin = [](const char* name) {
      ynn::perfetto_session::global()->begin(name);
      return reinterpret_cast<slinky::index_t>(name);
    };
    eval_config.trace_end = [](slinky::index_t token) {
      ynn::perfetto_session::global()->end();
    };
  }
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
  if (ynn_traceme_enabled()) {
    if (ynn::perfetto_session::global()) {
      YNN_LOG_WARNING()
          << "tsl::profiler tracing is overriding perfetto tracing.";
    }
    eval_config.trace_begin = [](const char* name) {
      return static_cast<slinky::index_t>(
          tsl::profiler::TraceMe::ActivityStart(name));
    };
    eval_config.trace_end = [](slinky::index_t token) {
      tsl::profiler::TraceMe::ActivityEnd(token);
    };
  }
#endif
  eval_context.config = &eval_config;

  values.reserve(subgraph->values.size());
  for (const ynn_value& i : subgraph->values) {
    values.push_back(ynn_runtime_value(i));
    ynn_runtime_value& value = values.back();
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (!value.symbol.defined()) {
      value.symbol = globals.symbols.insert_unique(value.name());
    }
    if (value.is_static()) {
      value.buffer =
          slinky::buffer_expr::make_constant(value.symbol, value.data);
    } else if (value.is_external()) {
      value.make_buffer(*this);

      for (size_t d = 0; d < value.extents.size(); ++d) {
        slinky::expr extent_d = i.physical_extent(d);
        if (!extent_d.defined()) {
          value.buffer->dim(d).bounds = slinky::point(0);
        } else if (const auto v = as_constant(extent_d)) {
          value.buffer->dim(d).bounds = slinky::range(0, *v);
        }
      }

      if (value.is_external_input()) {
        if (!value.data) {
          value.data =
              slinky::raw_buffer::make(i.rank(), ynn::type_size_bytes(i.type));
        } else {
          assert(value.data->rank == value.rank());
        }
      }
    }
  }
}

// This function takes the subgraph and turns it into a slinky pipeline.
ynn_status ynn_runtime::build() {
  std::vector<slinky::buffer_expr_ptr> inputs;
  std::vector<slinky::buffer_expr_ptr> outputs;
  funcs.clear();
  for (ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (value.is_static()) {
      assert(value.buffer);
      assert(value.buffer->constant());
    } else if (value.is_external_input()) {
      assert(value.buffer);
      inputs.push_back(value.buffer);
    }
  }

  for (const ynn_node& i : subgraph->nodes) {
    if (!i.is_valid()) continue;
    ynn_status status = i.create(i, *this);
    if (status != ynn_status_success) {
      return status;
    }

    for (uint32_t j : i.outputs) {
      ynn_runtime_value& value = values[j];
      if (!value.is_valid()) continue;
      assert(value.buffer->elem_size().defined());
      if (value.is_external_output() &&
          (!value.data || value.data->rank != value.rank())) {
        value.data = slinky::raw_buffer::make(
            value.rank(), *as_constant(value.buffer->elem_size()));
      }
    }
  }

  for (ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (value.is_external_output()) {
      assert(value.buffer);
      outputs.push_back(value.buffer);

      // This should be assert(value.data), but let's do that in a follow-up.
      if (value.data) {
        for (size_t d = 0; d < value.extents.size(); ++d) {
          if (!value.extents[d].defined() ||
              slinky::is_constant(value.extents[d], 1)) {
            value.data->mutable_dim(d) = slinky::dim::broadcast();
          }
        }
      }
    }
  }

  if ((flags & YNN_RUNTIME_FLAG_NO_SCHEDULE) == 0) {
    schedule();
  }

  slinky::build_options options;
  if ((flags & YNN_FLAG_ENABLE_SLINKY_TRACE) != 0) {
    options.trace = true;
  }
#ifdef YNN_ENABLE_PERFETTO
  options.trace = options.trace || get_trace_filename() != nullptr;
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
  options.trace = options.trace || ynn_traceme_enabled();
#endif
#ifdef NDEBUG
  options.no_checks = true;
#endif

  pipeline = slinky::build_pipeline(globals.symbols, {}, inputs, outputs,
                                    globals.lets, options);

  if (const char* ir_path = getenv("YNN_SCHED_IR")) {
    // Appends each built pipeline, so a process that builds several dumps all
    // of them; the autotuner hashes the file to recognize schedules it has
    // already benchmarked.
    static std::mutex ir_mutex;
    std::lock_guard<std::mutex> lock(ir_mutex);
    static std::ofstream ir_file(ir_path);
    slinky::print(ir_file, pipeline.body, &globals.symbols);
    ir_file.flush();
  }

  slinky::call_stmt::attributes attrs;
  attrs.name = "ynn_reshape_runtime";
  reshape_impl = slinky::let_stmt::make(
      globals.lets, slinky::call_stmt::make(make_reshape_impl(this), {}, {}, {},
                                            std::move(attrs)));
  return ynn_status_success;
}

ynn_status ynn_runtime::reshape() {
  setup();
  slinky::index_t result = slinky::evaluate(reshape_impl, eval_context);
  static_assert(ynn_status_success == 0, "");
  return static_cast<ynn_status>(result);
}

ynn_status ynn_runtime::setup() {
  std::vector<const slinky::raw_buffer*> inputs;
  std::vector<const slinky::raw_buffer*> outputs;
  for (const ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
    } else if (value.is_external_input()) {
      assert(value.data);
      inputs.push_back(value.data.get());
    } else if (value.is_external_output()) {
      assert(value.data);
      outputs.push_back(value.data.get());
    }
  }

  pipeline.setup(inputs, outputs, eval_context);
  return ynn_status_success;
}

ynn_status ynn_create_runtime(ynn_subgraph_t subgraph,
                              ynn_threadpool_t threadpool, uint32_t flags,
                              ynn_runtime_t* runtime_out) {
  YNN_RETURN_IF_ERROR(ynn::validate_subgraph("create_runtime", subgraph));
  if (runtime_out == nullptr) {
    YNN_LOG_ERROR() << "runtime_out must be non-null";
    return ynn_status_invalid_parameter;
  }

  slinky::thread_pool* slinky_threadpool =
      reinterpret_cast<slinky::thread_pool*>(threadpool);
  auto runtime =
      std::make_unique<ynn_runtime>(subgraph, slinky_threadpool, flags);
  YNN_RETURN_IF_ERROR(runtime->build());
  YNN_RETURN_IF_ERROR(runtime->setup());

  *runtime_out = runtime.release();
  return ynn_status_success;
}

ynn_status ynn_update_runtime_with_threadpool(ynn_runtime_t runtime,
                                              ynn_threadpool_t threadpool) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  runtime->eval_config.thread_pool =
      reinterpret_cast<slinky::thread_pool*>(threadpool);
  return ynn_status_success;
}

ynn_status ynn_runtime::invoke() {
  if (!pipeline.body.defined()) {
    // This pipeline is a no-op.
    return ynn_status_success;
  }
  return pipeline.evaluate(eval_context) ? ynn_status_error
                                         : ynn_status_success;
}

ynn_status ynn_set_external_value_shape(ynn_runtime_t runtime,
                                        uint32_t external_id, size_t rank,
                                        const size_t* dims) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  ynn_runtime_value& value = runtime->value(external_id);
  return value.set_external_shape(rank, dims);
}

ynn_status ynn_get_external_value_shape(ynn_runtime_t runtime,
                                        uint32_t external_id, size_t* rank,
                                        size_t* dims) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  const ynn_runtime_value& value = runtime->value(external_id);
  return value.get_external_shape(rank, dims);
}

ynn_status ynn_reshape_runtime(ynn_runtime_t runtime) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  return runtime->reshape();
}

ynn_status ynn_set_external_value_data(ynn_runtime_t runtime,
                                       uint32_t external_id, void* data) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  ynn_value& value = runtime->values[external_id];
  assert(value.data);
  value.data->base = data;
  return ynn_status_success;
}

ynn_status ynn_invoke_runtime(ynn_runtime_t runtime) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  return runtime->invoke();
}
namespace {

int32_t get_max_concurrency(const ynn_runtime& runtime) {
  // Traverse the pipeline body for any loops. If we find a parallel loop, we
  // return `max_int32`. Otherwise, we return 1.
  class visitor : public slinky::recursive_node_visitor {
   public:
    explicit visitor(const ynn_runtime& runtime) : runtime_(runtime) {}

    int32_t result = 1;
    void visit(const slinky::loop* op) override {
      if (!slinky::prove_true(op->max_workers == 1,
                              runtime_.globals.fact_bounds,
                              runtime_.globals.fact_alignment)) {
        result = std::numeric_limits<int32_t>::max();
      }
      slinky::recursive_node_visitor::visit(op);
    }

   private:
    const ynn_runtime& runtime_;
  } v(runtime);
  if (runtime.pipeline.body.defined()) {
    runtime.pipeline.body.accept(&v);
  }
  return v.result;
}

}  // namespace

ynn_status ynn_query_runtime(ynn_runtime_t runtime,
                             enum ynn_runtime_property property, void* result,
                             size_t* result_size) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!result_size || !result) {
    YNN_LOG_ERROR() << "result and result_size must be non-null";
    return ynn_status_invalid_parameter;
  }

  switch (property) {
    case ynn_runtime_property_concurrency: {
      memset(result, 0, *result_size);
      if (*result_size < sizeof(int32_t)) {
        YNN_LOG_ERROR() << "result must be an int32_t.";
        return ynn_status_error;
      }
      *result_size = sizeof(int32_t);
      int32_t max_threads = get_max_concurrency(*runtime);
      memcpy(result, &max_threads, sizeof(int32_t));
      return ynn_status_success;
    }
  }
  YNN_LOG_ERROR() << "Unknown runtime property: " << property;
  return ynn_status_error;
}

void ynn_delete_runtime(ynn_runtime_t runtime) { delete runtime; }

}  // extern "C"
