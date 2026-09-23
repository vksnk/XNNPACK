# Scheduler autotuner + un-fuse rule: handoff (2026-09-23)

Experiment scripts and notes for the schedule autotuner and the reduce
"un-fuse" rule. Not meant to land; this directory exists so the work can be
resumed.

## Branches

| Branch | What |
| --- | --- |
| `vksnk/ynnpack-autotune` | master (5de8644929) + sub-byte alignment fix + static-shapes shim knob + schedule tuner (`YNN_SCHED_TUNE`) + `autotune.py` (wall + CPU time) + opt-in step clamp (`YNN_STEP_CLAMP=1`) + un-fuse rule (two-pass cost model, 20951aa3f7) + this directory. |
| `vksnk/ynnpack-unfuse-rule` | Candidate for landing: master + sub-byte fix (329a84208a) + `YNN_XNNPACK_STATIC_SHAPES` knob (2859ca992b) + ONE tuner-less rule commit (d2a1a2e10d, +494 lines in `runtime.cc`). Generated from the autotune branch's runtime.cc with `strip_tuner.py`. |
| `vksnk/ynnpack-subbyte-step-alignment` | The sub-byte crop-alignment fix alone (found by random schedules). |

The rule still has env knobs (`YNN_UNFUSE=0`, `YNN_UNFUSE_BUDGET`,
`YNN_UNFUSE_TILE`, `YNN_UNFUSE_DEBUG=1`, `YNN_XNNPACK_STATIC_SHAPES`) — they
must be removed before landing.

## The un-fuse rule (current form)

Problem: a reduction-rooted fused chain (blockwise dot -> scale -> cross-block
reduce_sum) runs all the dot work under a serial k-block loop; at decode sizes
the nest has fewer tasks than threads. Materializing the pre-reduction
intermediate (compute_root) and parallelizing it wins 1.1-2.4x.

`runtime.cc`, in the fusion walk of `ynn_runtime::schedule()`:

1. **Candidate**: pure elementwise producer (no required splits, no own
   reduction splits) of a nest whose task lower bound is < threads, with its
   provable tile bytes <= budget (24MB, L3-ish).
2. **Cost model** `unfuse_pays(row_bytes, in_bytes, run_bytes, tasks, threads)`:
   `t_fused = traffic/(BW*p)` vs
   `t_unfused = traffic/(BW*T) + row/(BW_reread*T) + SYNC`, with
   `BW = 30 B/ns/core`, `SYNC = 3us`, `latency = 100ns`
   (`BW_reread = min(BW, run_bytes/latency)`: short runs are latency-bound).
   The old fitted gates (ratio >= 2T, 16KB/thread floor, 512B contiguous run)
   all fall out of these estimates. Constants are 9900X values; they are meant
   to be per-target parameters.
3. **Two-pass**: pass 1 runs the walk with the rule off and records static
   candidates; each is decided against the FINISHED nest's task count
   (`nest_tasks_lower_bound`, tight for static shapes); splits are restored and
   pass 2 re-runs the walk applying the decisions. Needed because nests
   self-heal later in the walk (bs256: the dot's required n-split retiles
   out3.d0 1024->256, 1->4 tasks) — no mid-walk bound can see that.
4. **Symbolic-batch arm** (dynamic shapes, e.g. through the XNNPACK shim
   without `YNN_XNNPACK_STATIC_SHAPES`): fuse only through batch loops with
   PINNED constant tile steps (`loop_level::step_pinned`, honored by
   `reconcile_step`). ~85 of the 473 rule lines; droppable if static dims
   reach the scheduler (see "Open questions").

Fire action: compute_at = batch-prefix depth (0 = root), refine the
function's whole-extent splits to `ceil(ub / (4*threads))`, propose step/4 on
the remaining nest loops.

## Results (9900X, single-CCD taskset, 4 threads)

- Qwen3-0.6B blockwise matrix (`qwen2/matrix.py`; m in {1,2,4,512} x 5 (n,k)
  pairs x {int4 bs32, int8 bs32, int4 bs256} + lm_head): decode m=1/2/4 wins
  1.1-2.4x (bs32), bs256 neutral (the fitted-gate version regressed it to
  0.57-0.61x at k>=2048), prefill m=512 and lm_head neutral (+-2%).
- Decode fires cost 1.6-2.8x CPU (fixed ~10-15us sync/wakeup per fire). Open
  policy question: latency vs energy on mobile decode.
- Dynamic (symbolic arm) QB4W fully_connected: 1.4-2.3x for M=1..256; M=512
  pays ~1.4x (intermediate grows with m).
- All ynnpack test suites pass with the rule on, including random schedules.

## Calibration campaigns (stock scheduler, rule off)

- **Softmax** (`sm/`, FP32SoftmaxDecomp M128/N256/K512, masks 1-7): tuner
  space exhausted, best <=1.05x (and those cost CPU). Outer-axis masks are
  3.2x slower than siblings; the reduce kernels already have both loop orders
  and dispatch correctly — the residual per-pass penalty at DRAM-resident
  64MB is unattributed (needs perf counters). Biggest prize for all masks:
  fusing the 5-pass decomposition (~3.4 -> ~2ms roofline).
- **LayerNorm** (`ln/`): mask 3 1.22x (two stat nests get one task each ->
  step /4); mask 7 1.13x (depth-1 materialization, `v7.fuse=1`; root is 1.6x
  worse). Step clamp prototype (`YNN_STEP_CLAMP=1`, functions with exactly one
  pure dim -> step <= ceil(extent/(2*threads))) gets mask 3 to 1.19x;
  universal variants regress (break fusion / neuter the un-fuse rule).

## Open questions / next steps

1. **Static-only simplification**: if the XNNPACK shim forwards real dims
   (make `YNN_XNNPACK_STATIC_SHAPES` behavior default), drop the symbolic arm
   (~85 lines: prefix walk, pinning, `step_pinned`, `unfuse_batch_tile`). If
   the shim keeps erasing dims, that arm is the only thing firing in litert-lm.
2. Remove env knobs; make the cost-model constants per-target.
3. CPU/energy policy for decode fires.
4. Depth-1 materialization (constant outer serial loops as a materialization
   prefix) — LayerNorm mask 7 class.
5. Step clamp needs a gate for latency-bound stat nests (+26% CPU at 1.00x
   wall on softmax mask 3) before default-on.
6. Softmax outer-axis residual attribution; fusing the decomposition chain.

## Methodology (important)

- One case per process; interleaved A/B, min-of-N; canary guard before each
  batch (bs256 m=1 < 8us = healthy machine).
- Pin to one CCD (`taskset -c 0-5` or `6-11`). Bandwidth-bound benches
  (softmax, LayerNorm) must run on ONE lane — DDR is socket-shared and a second
  lane doubled baselines. Compute-bound blockwise searches tolerate two lanes.
- `bench/subgraph/*` MUST be built with `--define=xnnpack_use_ynnpack=true`,
  otherwise it silently benchmarks stock XNNPACK (tell: ~10x faster times,
  PeakAllocated_MB != 0).
- `ynnpack/composites:dot_bench`: `--benchmarks=blockwise -m=... -n=... -k=...
  --thread_count=4` plus google-benchmark flags; static shapes natively.
- Autotuner: `YNN_SCHED_TUNE=record:<space>` then
  `python3 ynnpack/tools/autotune.py random|hillclimb|report ...`; use
  `--benchmark_min_time>=100x --benchmark_repetitions=3`. Results carry
  `[real_ns, cpu_ns]`.

## Scripts

Recovered from the session scratchpad; paths assume `~/Projects/XNNPACK` and
write results next to the script.

- `qwen2/matrix.py`, `qwen2/compare.py`: Qwen3-0.6B rule-off/on matrix and
  three-way comparison.
- `qwen/campaign.py`, `qwen/run_case.sh`: per-shape tuner search campaigns.
- `dyn/dyn_search.py`: dynamic-shape search on fully_connected.
- `cpu/*.py`: CPU-time A/B, m=1 intermediate-size boundary, work-floor
  validation.
- `sm/`, `ln/`: softmax / LayerNorm baselines, campaigns, validation, minimal
  decision sets, step-clamp battery.
- `strip_tuner.py <full runtime.cc> <out>`: regenerates the tuner-less
  runtime.cc for `vksnk/ynnpack-unfuse-rule` (asserts no "tuner" remains).
