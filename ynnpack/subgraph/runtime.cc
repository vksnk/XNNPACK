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
#include <functional>
#include <limits>
#include <map>
#include <memory>
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
    ynn::span<const slinky::var> dims, ynn::schedule_params params) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }
  params.dims.assign(dims.begin(), dims.end());

  // The loop splits are built and their steps computed by schedule() (see
  // compute_declared_steps), which runs before matching and can use global
  // (cross-function) context to adjust the declaration first.
  auto sched = std::make_unique<ynn::scheduling_info>();
  sched->params = std::move(params);
  return sched;
}

void ynn_runtime::compute_declared_steps(ynn::scheduling_info& sched) {
  const ynn::schedule_params& params = sched.params;
  const int rank = params.dims.size();

  std::vector<slinky::expr> splits =
      make_split_factors(globals, params.extents, params.element_cost,
                         params.given_splits, params.loop_order,
                         params.alignments);

  // Build the loop splits in the declared loop order. A given split that is
  // undefined means "no loop for this dimension".
  auto get_loop_dim = [&](int index_d) {
    return index_d < params.loop_order.size() ? params.loop_order[index_d]
                                              : index_d;
  };
  std::vector<ynn::scheduling_split> loop_splits;
  for (int index_d = 0; index_d < rank; ++index_d) {
    int d = get_loop_dim(index_d);
    if (d >= params.extents.size() || !params.extents[d].defined()) continue;
    if (d < params.given_splits.size() && !params.given_splits[d].defined()) {
      continue;
    }
    const bool required =
        d < params.step_required.size() && params.step_required[d];
    loop_splits.push_back(
        {params.dims[d], splits[d], params.extents[d], required});
  }
  sched.loop_splits = std::move(loop_splits);
}

namespace {

// Just a helper structure to track information about loop levels.
struct loop_level {
  slinky::loop_id loop_id;
  slinky::expr extent;
  slinky::expr step;
  bool step_is_required = false;
  // The index of the parent loop in the global loop nest, or -1 for the
  // outermost loops. Loops are appended after their parent, so the parent
  // index is always less than the index of the loop itself.
  int parent = -1;
  // The number of workers the loop should use, computed by compute_workers()
  // once the whole nest is built.
  slinky::expr workers = slinky::loop::serial;
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
  if (slinky::prove_true(a == b)) return a;
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
  // Enough tasks to have good load balancing.
  const slinky::index_t target_task_count =
      max_threads > 1 ? max_threads * 2 : 1;

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
    // A loop whose step provably covers its extent runs exactly one
    // iteration, so it is identical for any number of functions inside it
    // (required steps are excluded). The proof needs the global lets resolved
    // -- split factors are frequently `min(...)` expressions that the
    // simplifier already reduced to the extent itself, but hidden behind a let
    // variable slinky can't see through when it builds the loop. Replacing the
    // step with the extent expression lets slinky prove the single iteration
    // and fold the loop away entirely.
    const bool elide_allowed =
        !l.step_is_required && globals.is_pure_dim(l.loop_id.var);
    // Serial reduction ("k") dims additionally qualify for the
    // single-iteration elision below (but not for the widening elisions):
    // with provably one iteration there is no accumulation blocking to
    // preserve.
    const bool single_iteration_elide_allowed =
        elide_allowed ||
        (!l.step_is_required && globals.is_reduction_dim(l.loop_id.var));
    const slinky::expr simplified_extent = single_iteration_elide_allowed
                                               ? slinky::simplify(l.extent)
                                               : slinky::expr();
    if (single_iteration_elide_allowed &&
        slinky::prove_true(simplified_extent <=
                           resolve_let_var(globals, l.step))) {
      l.step = slinky::max(simplified_extent, 1);
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

// Reconciles the loop steps of fused chains rooted at reductions. A
// reduction under footprint pressure (the fused chain's live set exceeds the
// cache-derived budget) wants its reduction loops innermost -- the
// accumulator tile stays resident across the whole reduction, and with the
// parallel pure loops outside there are no per-step thread pool joins --
// with one consistently tiled pure loop shared by every member of the
// chain. Measured on the fused blockwise dot (m=256/n=4096/k=4096 bs=32,
// 9900X): 18.7 -> 16.4ms at 1 thread and 4.3 -> 2.4ms at 8 threads.
//
// The decision is made once, at the root: the members' declarations are
// rewritten so the whole chain declares the same steps and matching needs no
// overrides. Chains that contain a function with required steps that
// provably differ from the planned steps are left alone -- imposing the
// chain's tiles would unfuse that member (e.g. attention's QK dot feeding
// softmax), and recomputing the materialized intermediate costs far more
// than the reordering wins.

// A function's defined dimensions, collected once per function so the chain
// passes don't repeat the positional guards of schedule_params.
struct chain_dim {
  int d;  // Index into the schedule_params vectors.
  slinky::expr extent;
  bool pure;
  int region;  // Inferred source region, or -1 if unknown.
  std::optional<slinky::index_t> size;  // Constant extent, if known.
};

constexpr slinky::index_t chain_budget_elems = (1 << 20) / 16;

// The budget-derived tile for dims[first], sized so that the fused body's
// live set -- roughly tile * product of the other dimensions -- stays within
// the budget. Returns nothing when there is no footprint pressure: the
// extents are not all constant, there are no other dimensions to keep the
// outer loops busy, or the tile covers the dimension whole.
std::optional<slinky::index_t> footprint_tile(
    ynn::span<const chain_dim> dims, size_t first) {
  if (!dims[first].size) return std::nullopt;
  slinky::index_t other = 1;
  for (size_t j = 0; j < dims.size(); ++j) {
    if (j == first) continue;
    if (!dims[j].size) return std::nullopt;
    other *= *dims[j].size;
  }
  if (other <= 1) return std::nullopt;
  const slinky::index_t tile =
      std::max<slinky::index_t>(64, chain_budget_elems / other);
  if (tile >= *dims[first].size) return std::nullopt;
  return tile;
}

slinky::expr make_tile_step(slinky::index_t tile, const slinky::expr& extent) {
  return slinky::simplify(slinky::min(tile, slinky::max(extent, 1)));
}

void reconcile_chains(
    ynn::slinky_globals& globals, std::vector<slinky::func>& funcs,
    const std::map<std::pair<slinky::var, int>, int>& source_regions) {
  auto sched_of = [&](int i) {
    return static_cast<ynn::scheduling_info*>(funcs[i].user_data());
  };

  auto dims_of = [&](int i) {
    const ynn::schedule_params& p = sched_of(i)->params;
    std::vector<chain_dim> dims;
    for (int d = 0; d < static_cast<int>(p.dims.size()); ++d) {
      if (d >= static_cast<int>(p.extents.size()) || !p.extents[d].defined()) {
        continue;
      }
      auto [buf, buf_dim] = find_output_dim(&funcs[i], p.dims[d]);
      auto it = source_regions.find(std::make_pair(buf, buf_dim));
      dims.push_back({d, p.extents[d], globals.is_pure_dim(p.dims[d]),
                      it != source_regions.end() ? it->second : -1,
                      slinky::as_constant(p.extents[d])});
    }
    return dims;
  };

  std::map<slinky::var, int> producer_of;
  std::map<slinky::var, std::vector<int>> consumers_of;
  for (int i = 0; i < static_cast<int>(funcs.size()); ++i) {
    for (const auto& out : funcs[i].outputs()) {
      producer_of[out.sym()] = i;
    }
    for (const auto& in : funcs[i].inputs()) {
      consumers_of[in.sym()].push_back(i);
    }
  }

  // A chain member's steps can be freely rewritten: it declares its
  // scheduling inputs, chooses none of its splits itself (a partial
  // reduction's pr_split loops must stay as declared), requires none of its
  // steps, and is not pinned to root.
  auto is_member = [&](int i) {
    const ynn::scheduling_info* s = sched_of(i);
    if (!s || s->force_root || s->params.dims.empty()) return false;
    for (bool r : s->params.step_required) {
      if (r) return false;
    }
    for (const slinky::expr& g : s->params.given_splits) {
      if (g.defined()) return false;
    }
    return true;
  };

  // A function whose declaration includes reduction loops the scheduler
  // chooses itself; each is the root of its own chain decision.
  auto has_computed_reduction = [&](int i) {
    const ynn::scheduling_info* s = sched_of(i);
    if (!s) return false;
    const ynn::schedule_params& p = s->params;
    for (int d = static_cast<int>(p.given_splits.size());
         d < static_cast<int>(p.dims.size()); ++d) {
      if (d < static_cast<int>(p.extents.size()) && p.extents[d].defined() &&
          !globals.is_pure_dim(p.dims[d])) {
        return true;
      }
    }
    return false;
  };

  // Collects the chain of `r`: every member reachable through
  // producer/consumer edges between members. Other reductions are boundaries
  // -- each makes its own chain decision -- and functions with required
  // steps stop the walk and are recorded for the compatibility check.
  // Returns false if the chain overlaps one that was already reconciled.
  auto collect_chain = [&](int r, const std::vector<char>& claimed,
                           std::vector<int>& members,
                           std::vector<int>& required_boundary) {
    std::vector<char> visited(funcs.size(), 0);
    std::vector<int> worklist = {r};
    visited[r] = 1;
    bool ok = true;
    auto visit = [&](int j) {
      if (visited[j]) return;
      visited[j] = 1;
      if (j != r && has_computed_reduction(j)) return;
      if (claimed[j]) {
        // Already decided by another chain; do not re-tile it.
        ok = false;
      } else if (is_member(j)) {
        worklist.push_back(j);
      } else {
        const ynn::scheduling_info* s = sched_of(j);
        if (s && !s->params.step_required.empty()) {
          required_boundary.push_back(j);
        }
      }
    };
    while (!worklist.empty() && ok) {
      int i = worklist.back();
      worklist.pop_back();
      members.push_back(i);
      for (const auto& in : funcs[i].inputs()) {
        auto it = producer_of.find(in.sym());
        if (it != producer_of.end()) visit(it->second);
      }
      for (const auto& out : funcs[i].outputs()) {
        auto it = consumers_of.find(out.sym());
        if (it == consumers_of.end()) continue;
        for (int j : it->second) visit(j);
      }
    }
    return ok;
  };

  std::vector<char> claimed(funcs.size(), 0);
  for (int r = 0; r < static_cast<int>(funcs.size()); ++r) {
    // A root is a reduction whose reduction loops the scheduler chooses
    // itself (a partial reduction's given pr_split loops are pure and stay
    // as declared).
    if (!is_member(r) || claimed[r] || !has_computed_reduction(r)) continue;

    // The chain's plan: the root's first pure dimension gets the footprint
    // tile, its remaining pure dimensions stay whole. Everything is keyed by
    // source region, which is how the matcher identifies shared loops.
    std::vector<chain_dim> root_dims = dims_of(r);
    std::vector<chain_dim> root_pure;
    for (const chain_dim& dim : root_dims) {
      if (dim.pure) root_pure.push_back(dim);
    }
    if (root_pure.empty() || root_pure[0].region == -1) continue;
    std::optional<slinky::index_t> tile = footprint_tile(root_pure, 0);
    if (!tile) continue;
    std::map<int, slinky::expr> planned;
    planned[root_pure[0].region] = make_tile_step(*tile, root_pure[0].extent);
    for (size_t j = 1; j < root_pure.size(); ++j) {
      if (root_pure[j].region != -1) {
        planned.emplace(root_pure[j].region, root_pure[j].extent);
      }
    }

    std::vector<int> members;
    std::vector<int> required_boundary;
    if (!collect_chain(r, claimed, members, required_boundary)) continue;

    // Compatibility check: a boundary function's required steps must
    // provably match the planned steps of the loops it shares with the
    // chain, or fusing it would fail and materialize its output.
    bool ok = true;
    for (int q : required_boundary) {
      const ynn::schedule_params& qp = sched_of(q)->params;
      for (const chain_dim& dim : dims_of(q)) {
        if (dim.d >= static_cast<int>(qp.step_required.size()) ||
            !qp.step_required[dim.d] ||
            dim.d >= static_cast<int>(qp.given_splits.size()) ||
            !qp.given_splits[dim.d].defined()) {
          continue;
        }
        auto it = planned.find(dim.region);
        if (it != planned.end() &&
            !slinky::prove_true(qp.given_splits[dim.d] == it->second)) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    if (!ok) continue;

    // Rewrite the members' declarations so the whole chain declares the
    // planned steps. Dimensions shared with the chain's loops get the
    // planned steps (for the root's own pure dimensions this is the plan
    // itself). A dimension of a member's own (e.g. the quantized row
    // dimension of the dynamic quantization prologue, which the chain's
    // loops never cover) gets a footprint tile of its own for the first
    // such dimension -- the member may root its own nest there, and a whole
    // extent would serialize it -- and whole extents for the rest.
    for (int i : members) {
      claimed[i] = 1;
      ynn::schedule_params& mp = sched_of(i)->params;
      std::vector<chain_dim> dims = dims_of(i);
      std::vector<slinky::expr> given(mp.dims.size());
      std::vector<slinky::expr> defaults;
      if (i == r) {
        // The root's reduction loops go innermost (pure loops outer). Their
        // steps keep the computed defaults, raised to pairs when provably 1
        // so the per-step call overhead is amortized -- never lowered.
        defaults = make_split_factors(globals, mp.extents, mp.element_cost,
                                      mp.given_splits, mp.loop_order,
                                      mp.alignments);
        std::vector<int> order;
        order.reserve(dims.size());
        for (const chain_dim& dim : dims) {
          if (!dim.pure) order.push_back(dim.d);
        }
        for (const chain_dim& dim : dims) {
          if (dim.pure) order.push_back(dim.d);
        }
        mp.loop_order = std::move(order);
      }
      std::vector<chain_dim> pure;
      for (const chain_dim& dim : dims) {
        if (dim.pure) pure.push_back(dim);
      }
      bool first_unshared = true;
      for (size_t j = 0; j < dims.size(); ++j) {
        const chain_dim& dim = dims[j];
        if (!dim.pure) {
          slinky::expr split = defaults[dim.d];
          if (auto c = slinky::as_constant(split)) {
            split = slinky::simplify(slinky::max(
                slinky::expr(*c), slinky::min(2, slinky::max(dim.extent, 1))));
          }
          given[dim.d] = split;
          continue;
        }
        auto it = planned.find(dim.region);
        if (it != planned.end()) {
          given[dim.d] = it->second;
          continue;
        }
        given[dim.d] = dim.extent;
        if (first_unshared) {
          first_unshared = false;
          size_t pure_j = 0;
          while (pure[pure_j].d != dim.d) ++pure_j;
          if (std::optional<slinky::index_t> local =
                  footprint_tile(pure, pure_j)) {
            given[dim.d] = make_tile_step(*local, dim.extent);
          }
        }
      }
      mp.given_splits = std::move(given);
    }
  }
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
  // Maps {buffer_sym, dim_index} to its inferred source region unique
  // identifier.
  std::map<std::pair<slinky::var, int>, int> source_regions =
      infer_source_regions(funcs);

  // Rewrite the declarations of reduction-rooted fused chains so their
  // members declare consistent steps, then compute the steps of the
  // declared (not given) splits. Functions declare their scheduling inputs
  // at create time (see schedule_params); the steps are computed here, in
  // creation order.
  reconcile_chains(globals, funcs, source_regions);
  for (slinky::func& f : funcs) {
    auto* sched = static_cast<ynn::scheduling_info*>(f.user_data());
    if (sched && !sched->params.dims.empty()) {
      compute_declared_steps(*sched);
    }
  }

  // This a list of indices of consumers of a given buffer.
  std::map<slinky::var, std::vector<int>> consumers;
  // This is a tree representing a global loop nest of a whole pipeline so
  // far. For efficiency and convenience, it's stored as an array of nodes
  // with auxiliary structures using indices to refer to the loop levels.
  std::vector<loop_level> global_loop_nest;

  std::vector<scheduling_data> func_scheduling_data(funcs.size());

  auto get_source_region = [&](slinky::var buf, int dim) {
    auto key = std::make_pair(buf, dim);
    auto it = source_regions.find(key);
    return it != source_regions.end() ? it->second : -1;
  };

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
        if (prove_true(loop_splits[split_i].extent == 1)) {
          split_matched[split_i] = true;
        }
      }

      // Walk the loop nest from the outermost loop inwards. For each loop,
      // find a split of this function which covers the same source region.
      // The splits don't have to be matched in their declared order: loops
      // over pure dims carry no state across iterations, so they can be
      // freely reordered, and splits which were not matched simply remain the
      // function's own inner loops. Non-pure (reduction) splits do carry
      // state across iterations (they accumulate into the same output), so
      // they act as a fence: nothing is matched at or beyond the first one.
      // We must stop at the first loop of the nest we can't cover: computing
      // the function inside a loop which doesn't slice its output would
      // recompute the function on every iteration of that loop.
      compute_at = 0;
      while (compute_at < loop_nest.size()) {
        loop_level& global_loop = global_loop_nest[loop_nest[compute_at]];
        // Map the consumer's loop variable back to its output dimension
        // index.
        auto [consumer_buf, consumer_dim] =
            find_output_dim(global_loop.loop_id.func, global_loop.loop_id.var);
        const int consumer_source_region =
            consumer_dim != -1 && consumer_buf.defined()
                ? get_source_region(consumer_buf, consumer_dim)
                : -1;

        int matched_split = -1;
        if (consumer_source_region != -1) {
          // Whether the search has passed over an unmatched (and non-trivial)
          // split, i.e. matching a later split would reorder the function's
          // loops.
          bool out_of_order = false;
          for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
            if (split_matched[split_i]) continue;
            const ynn::scheduling_split& split = loop_splits[split_i];
            if (!globals.is_pure_dim(split.var)) {
              // We don't want to fuse a reduction dimension because it is
              // likely being broadcasted here, and we don't reorder other
              // splits across it either.
              break;
            }
            if (split.step_is_required && out_of_order &&
                !(global_loop.step.defined() &&
                  slinky::prove_true(split.step == global_loop.step))) {
              // A required step means the function deliberately chose the
              // blocking of this loop, and the loop order is likely a part of
              // the same deliberate choice. Matching it out of order would
              // impose that blocking on a nest built for a different order
              // (e.g. pull a dot under the loops of its elementwise consumer,
              // overriding the consumer's steps with the dot's tiles), so we
              // only allow such splits to be matched in their declared order —
              // unless the loop's step is provably already the required step,
              // in which case the match imposes nothing on the nest.
              continue;
            }
            // Map the producer's loop variable back to its output dimension
            // index.
            auto [producer_buf, producer_dim] = find_output_dim(&f, split.var);

            // Instead of comparing forward extents (which causes false
            // positives for unrelated constant extents), we check if both
            // loops share the exact same inferred source region identifier.
            if (producer_dim != -1 && producer_buf.defined() &&
                get_source_region(producer_buf, producer_dim) ==
                    consumer_source_region) {
              matched_split = split_i;
              break;
            }
            out_of_order = true;
          }
        }

        if (matched_split == -1) {
          break;
        }
        split_matched[matched_split] = true;

        const ynn::scheduling_split& split = loop_splits[matched_split];
        if (split.step_is_required) {
          if (global_loop.step_is_required &&
              !prove_true(split.step == global_loop.step)) {
            // Two producers require different tiles for this shared loop (e.g.
            // the two attention matmuls pick different query tiles). Use their
            // least common multiple so the loop is an integer number of *both*
            // tiles, keeping it a multiple of each kernel's m/n block. If the
            // LCM overflows, it clamps at max index_t (assuming no splitting).
            global_loop.step = lcm_sat(globals, global_loop.step, split.step);
          } else {
            if (std::optional<slinky::var> v =
                    slinky::as_variable(global_loop.step)) {
              // This is a special variable which defines partial reduction
              // bounds, so we need to override to match the loop step.
              if (globals.symbols.name(*v).rfind("pr_split", 0) == 0) {
                globals.update_let(*v, split.step);
              }
            }
            global_loop.step = split.step;
          }
          global_loop.step_is_required = true;
        }
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
                                    parent});
        loop_nest.push_back(global_loop_nest.size() - 1);
      }
    }

    // Record which buffers this function is consuming.
    for (const auto& input : f.inputs()) {
      consumers[input.buffer->sym()].push_back(i);
    }
  }

  // `thread_count()` reports the number of background worker threads. The
  // thread that invokes the runtime also participates as a worker (it runs
  // tasks while waiting in `thread_pool::wait_for`), so the effective
  // parallelism is one more than the reported count. Without this `+ 1`, a
  // pool with a single background thread (two threads of execution in total)
  // would be scheduled serially, and every other size would be sized one
  // worker short.
  const int max_threads = threadpool() ? threadpool()->thread_count() + 1 : 1;
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
    int32_t result = 1;
    void visit(const slinky::loop* op) override {
      if (!slinky::prove_true(op->max_workers == 1)) {
        result = std::numeric_limits<int32_t>::max();
      }
      slinky::recursive_node_visitor::visit(op);
    }
  } v;
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
