# PR16 2DGS backward speed and accuracy evaluation

## Verdict

PR16's combined tail-trim + reduce-scatter candidate is slightly faster in some
detached-raster workloads, but **it does not currently pass the real-scene full
gradient accuracy gate**. The `grad/densify` comparison is unstable even for
baseline versus baseline, and fails for both PR16 candidates. Therefore these
timings are diagnostic only; this is **not a validated 2DGS speedup**, and PR16
should remain Draft until the densification ordering/gradient issue is resolved
or a sound deterministic contract is established. Do not waive the PR9
`1e-4` gradient threshold.

## Exact variants

Fresh isolated worktrees/builds from the same RTX4090/CUDA/PyTorch environment:

| Variant | Commit | Change |
|---|---|---|
| Base | `9b331bc0d2f94c415569488d596e1ea4e783d950` | PR16 base |
| Tail | `39f4be60e3283739929e9cf843593a05d10eef70` | Tile suffix trimming only |
| Head | `3aeedd6ceb4907513237b4a515f53f1c8167866c` | Tail trimming + narrow reduce-scatter |

Binary SHA-256 hashes and workload settings are in
[`pr16_2dgs_evaluation.json`](pr16_2dgs_evaluation.json). Builds are isolated
under `/tmp/opencode/gsplat-pr16-eval`; the original user checkout and parent
repository were not edited.

## Workload and protocol

- Input: all **1,932,132** Gaussians from the supplied water2 PLY and
  COLMAP-derived cameras; views 0, 623 and 1247 at native 1280×720 and 3840×2160.
- The PLY was trained as a 3DGS/PGSR-style checkpoint, but is deliberately fed
  through gsplat's **2DGS rasterization path** to test kernel performance, as
  requested. This is not evidence for a converged 2DGS-trained model.
- `packed=False`, `RGB+ED`, `absgrad=True`, compact-box sweep, tile16, float32.
  The PLY's three scales are passed through the 2DGS projection API; no
  checkpoint semantics are asserted beyond this raster-kernel benchmark.
- Full wrapper and detached raster scopes are reported separately. Each scope
  measures inference forward, training forward, backward, and combined via
  CUDA events. Upstream gradients are seeded identically from the PR16 harness.
- Three serial Latin-square jobs: base/tail/head, head/base/tail,
  tail/head/base; **20 warmups and 40 timed samples** per case. Reported ratios
  use `baseline / candidate` on the median of per-job means. Raw run JSON,
  command order, GPU telemetry, parity, and all timing samples are retained in
  `evidence/scene-jobs/` outside the repository.
- Gates were not relaxed: outputs/IDs max-abs `2e-5` (IDs exact), every gradient
  relative-L2 `1e-4`. All output tensors and IDs are compared; densification is
  included, not excluded.

## Speed results

Ratios below are geometric means across the three selected views; greater than
1 means faster. At this small margin, treat the timings as indicative, not as a
stable production claim.

| Candidate vs base | Resolution | Scope | Forward | Backward | Combined |
|---|---:|---|---:|---:|---:|
| Tail only | Native | Full | 0.998× | 1.003× | 1.001× |
| Tail only | Native | Raster | 0.998× | 1.006× | 1.004× |
| Tail only | 4K | Full | 0.999× | 1.002× | 1.001× |
| Tail only | 4K | Raster | 0.998× | 1.003× | 1.001× |
| Tail + scatter | Native | Full | 0.999× | 1.007× | 1.003× |
| Tail + scatter | Native | Raster | 0.997× | **1.026×** | **1.012×** |
| Tail + scatter | 4K | Full | 1.001× | 1.003× | 1.002× |
| Tail + scatter | 4K | Raster | 1.003× | **1.007×** | **1.005×** |

Tail trimming alone is effectively neutral. The head's raster-only backward
gain is about **2.6% native** and **0.7% at 4K**. Full-render combined changes
are about **+0.3% native** and **+0.2% at 4K**, near timing noise. No 2DGS
speedup should be claimed from these measurements until the accuracy failure is
resolved and the effect is confirmed over more views/jobs.

![Speed and accuracy summary](pr16_speed_accuracy.png)

The CSV tables are `pr16_2dgs_timings.csv` and
`pr16_2dgs_accuracy.csv`; compact machine-readable values and exact hashes are
in `pr16_2dgs_evaluation.json`.

## Accuracy: blocking finding

The **52/52 synthetic fixtures pass** for base, tail, and head, including
baseline-repeat, tile-tail, batch boundaries, packed/unpacked, auxiliary losses,
and densification. The separate scene experiment tells a different story:

- RGB+ED, alpha, normal, surface-normal, distortion, median, and IDs are
  bit-identical in all measured head comparisons.
- Every compared gradient **except `grad/densify`** passes. The maximum
  relative-L2 across all other gradients is `1.45e-6` (below `1e-4`).
- Baseline repeat control itself fails the densification gate in **24/36**
  job/case comparisons; maximum relative-L2 is `5.81e-3`. Its densification
  error median across the 36 comparisons is `4.98e-4`.
- Tail-only fails the densification gate in **36/36** comparisons; max
  relative-L2 `5.22e-3`.
- Combined head fails in **36/36** comparisons; max relative-L2 `1.17e-2`,
  median `4.22e-3`.

This aligns with the source-level concern: 2DGS ray-transform gradients are
accumulated by multiple warps/blocks, while the densification proxy reads that
buffer without a global completion/order guarantee. A warp-level synchronization
cannot order contributions from other warps/blocks. Baseline instability does
not excuse candidate failures; no parity samples or gradients are dropped.

**Conclusion:** PR16 has not demonstrated accepted full-scene gradient
correctness. Keep the PR Draft and do not merge until densification semantics
are made deterministic/defined or the optimization is redesigned so the existing
strict gate passes. The small backward timing gain is not sufficient to waive
this blocker.

## Additional checks

- `tests/test_2dgs.py`: **17 passed** on the PR16 head (one existing meshgrid
  warning).
- PR16's 52-fixture suite: base self-check, base repeat, tail, and head all
  report pass.
- Compute Sanitizer on head: memcheck **0 errors**, synccheck **0 errors**,
  racecheck **0 hazards / 0 errors / 0 warnings** on the 52-case matrix.
- The known densification nondeterminism is a semantic/inter-kernel ordering
  issue; these raster sanitizer fixtures do not prove the real-scene densify
  proxy is race-free.

Reproduce with the commands in the PR body. Full reports are at
`/tmp/opencode/gsplat-pr16-eval/evidence/`; no PLY, reference tensors, binaries,
or GPU logs are added to the PR.
