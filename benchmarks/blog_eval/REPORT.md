# 3DGS CUDA blog → gsplat: evaluation log

## Current result (2026-10-29)

**Delivery commit: `88513ea`**, including the partial-warp fallback. An extra
tile4 check caught zero gradients in the earlier `b96efa8` binary; that binary
is retired. All results in this executive summary were refreshed on the rebuilt
delivery binary. It passes **166 cases on each stream mode**, memcheck and
synccheck (0 errors), and fixed-16 mean-gradient checks in RGB and RGB+ED.

The retained implementation is **backward-only**: narrow-channel warp
reduce-scatter, occluded-tail trimming, and hardened narrow-channel alpha
predication. Scalar color accumulation, cooperative SH zeroing, and forward
AABB warp culling were implemented/tested and **reverted** after their gates.

Final paired RGB measurements on the complete 20.8M-point scene:

| Camera | Raster baseline → optimized | Raster speedup | Full-render baseline → optimized | Full-render speedup |
|---|---:|---:|---:|---:|
| 0 | 23.319 → 17.444 ms | 1.337× | 69.148 → 62.851 ms | 1.100× |
| 2552 | 3.269 → 1.984 ms | 1.648× | 40.360 → 39.728 ms | 1.016× |
| 5103 | 2.626 → 1.926 ms | 1.363× | 40.084 → 38.575 ms | 1.039× |

Each value is the median of two job medians, with 40 warmups and 100 timed
iterations per job, alternating baseline/candidate. These are **three views,
not all 5,104 cameras**. See `results/REBUILT_FINAL_SUMMARY.md` for every pair.
Raster improvements repeat in every pair; full-render results remain
small/noisy, and camera 2552 regresses in one of its two pairs. No optimizer/loss/training-loop
speedup or article-level 7× result is claimed.

Final-binary full-scene RGB/alpha are bit-identical. Screen-space gradient
relative-L2 is at most **2.62e-6**; the 166-case matrix and both sanitizers pass.
**The strict raw-quaternion `1e-4` single-shot gate is not universally met:**
the rebuilt RGB and RGB+ED comparisons are both about `1.63e-4`. Fixed-16 mean
quaternion gradients pass (`4.57e-5` / `3.50e-5`) with unchanged thresholds, but
this does not turn either single-shot failure into a pass or prove convergence.

For the product wrapper's **RGB+ED + absgrad** settings on camera 0, three
paired jobs give ratios of job-median medians **1.311× raster-only / 1.081× full
differentiable rendering**. Forward image/alpha are exact and screen-space
gradients including absgrad pass (`≤1.48e-6` relative-L2). The raw quaternion
gate remains noise-limited (`1.63e-4`); this is not a strict full-parameter pass.

The implementation, report, harness and tutorial are maintained in this gsplat repository. This evaluation does not require a parent application checkout or parent gitlink change.

The following sections retain the chronological experiment log. Earlier
`b96efa8` checkpoints/tables are historical; `REBUILT_FINAL_SUMMARY.md`, the
partial-warp safety artifacts and the fixed-16 ensemble artifacts certify the
delivery binary. Earlier data are not pooled into final speedup estimates.

## Scope and environment

- Article: https://grgkopanas.github.io/3dgs_cuda_blog/
- gsplat baseline: `ffe2db3c378533767ccb1ad0d8707a3f33549c3e`.
- Python: use the Python environment configured for this gsplat checkout.
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, ~24 GiB.
- PyTorch: 2.10.0+cu128; CUDA 12.8. These versions are not changed.
- Scene: `2026-08-13_PDDtilted_1.dslam/outputs/f6d9613b`, iteration 030000.
- PLY: 20,789,814 Gaussians; degree-1 SH (3 DC + 9 rest coefficients).
- Real cameras: adjacent `cameras.json`, 1280×720. Export metadata also exists.
- Target: forward+backward acceleration, constrained by forward and gradient alignment.
- The supplied checkpoint's saved configuration names PGSR. This evaluation
  renders its PLY through gsplat's standard 3DGS API, as in the article; it does
  not measure PGSR-specific plane depth/normal kernels or training convergence.
- Initial timing tables use RGB and absgrad disabled to match the article's
  narrow RGB workload. The product wrapper defaults to RGB+ED and absgrad true;
  supplementary checks for those settings are being added, not assumed equal.

## Measurement rules

- Compare the same points, cameras, settings, dtype, and CUDA architecture.
- Separate the full differentiable rendering call (projection, SH, binning,
  rasterization and backward) from raster-only kernel timing when available.
- Exclude data loading and compilation; warm up before CUDA-event timing.
- Record sampled versus full-scene workloads explicitly; do not extrapolate
  subset speedups to all 20.8M points.
- Compare gradients with deterministic upstream image/alpha gradients and
  report relative L2 plus absolute errors, including baseline atomic noise.
- Preserve gsplat's 0.999 alpha cap and pixel-center/termination semantics,
  rather than adopting the article's different reference implementation.

## Follow-up: reference protocol and preflight ownership

After PR review, the benchmark protocol was made explicit and fail-fast:

- Single-pass snapshots use protocol **v3**; fixed-count mean-gradient snapshots
  retain the distinct `predeclared_raw_parameter_mean_v2` protocol.
- Reference compatibility compares SHA-256 input contents and declared workload
  settings. Absolute paths are recorded separately as local provenance.
- CPU preflight checks version, content/settings, then PLY/camera-derived point
  count, resolution, SH degree and required 3DGS property layout before loading
  the CUDA extension or uploading scene tensors. The parsed metadata is reused
  by tensor loading and actual shapes are checked against it.
- Standard 3DGS SH input is validated as contiguous `f_rest_0...`, three
  coefficients per RGB basis, square basis count, and degree 0--4 supported by
  this CUDA SH implementation. Invalid schemas fail with a clear `ValueError`.
- Existing PLY input property names/order, point sampling and tensor transforms
  are unchanged for valid files; the metadata pass reuses its memory-mapped
  `PlyData` object rather than parsing/uploading the scene twice.
- Reference save and compare are mutually exclusive, so a command cannot
  overwrite a baseline before loading and comparing it.

Old single-pass protocol-v2 and ensemble-v1 references must be regenerated; no
implicit conversion is attempted. Hashing the scene content is setup work outside
the timed rendering loop. CUDA device identity is only knowable after runtime
initialization; the ensemble checks it before its first rendering pass, while
file/workload/scene metadata incompatibilities fail before CUDA setup.

The review-follow-up candidate rebuilt successfully. It passed **166 differential
cases** on default and non-default streams, plus memcheck and synccheck (zero
errors each). CPU benchmark protocol tests and the standalone full-warp helper
contract test pass. These checks validate the contract refactor; no new performance
claim is inferred from them. The published timing table above remains the earlier
delivery-binary measurement because the follow-up changes signatures/layout
ownership but did not run a new timing study.

## Checkpoint 1 — source review and benchmark setup

The fork already supports opacity-aware compact-box intersection and analytic
sweep enumeration in `isect_tiles.cu`. `compact_box_impl="sweep"` is the API
default; enabling compact-box culling is still a separate flag. The product
3DGS wrapper already defaults that flag to true. Thus this is not a new
optimization that can be claimed as a fresh integration.

The backward rasterizer still uses component-wise full-warp reductions followed
by lane-0 atomic writes. Reduce-scatter is the first targeted new experiment.

An initial exploratory benchmark mistakenly passed log-scales directly to the
gsplat API. **Those exploratory timings and image comparisons are invalid and
are excluded from results.** The harness now exponentiates scales and applies
sigmoid to opacity. Camera matrices are inverse camera-to-world transforms.

The clean worktree lacked GLM headers. Its pinned dependency was initialized
without source edits or a revision change. No nested GLM code will be modified.
Building gsplat in place avoids replacing the user's installed gsplat package.

## Innovation status

| Technique | Status | Measured result |
|---|---|---|
| Opacity-aware row-span binning | Already in fork; ablated, not newly implemented | 16.05M → 9.52M pairs, camera 0 |
| Warp reduce-scatter | Retained | Initial isolated raster ablation 1.150× |
| Scalar backward color projection | Implemented, tested, reverted | No gain; 13.945 → 14.496 ms |
| SH streaming / zero elimination | Streaming mismatch assessed; selective zeroing tested/reverted | No reliable full-render gain |
| Contributor compaction / masks | Full compaction not ported; safe suffix trim retained | Initial incremental raster 1.223× |
| Branchless predication | Retained with explicit safe derivative/buffer gating | Final incremental camera-0 raster 1.038× |
| Packed splats / warp subtile masks | AABB-mask prototype tested/reverted; packed struct not ported | Small/noisy 1.020× raster result |
| Two-stage sorting | Source assessment only, not implemented | No measured result |
| Multirow / pipe math / BF16 SH | Source/hardware assessment only, not implemented | No measured result |

## Verified outcomes so far

## Experiment 1 — halving warp reduce-scatter

Implemented in `gsplat/cuda/include/warp_reduce_scatter.cuh` and the narrow-channel
backward path in `rasterize_to_pixels_bwd.cu` on gsplat branch
`perf/blog-cuda-speedups`. Channels >4 retain the previous implementation.
The first version includes optional absolute-mean gradients in the reduction
window; two-splat folding is not yet implemented.

**Full supplied scene fits:** 20,789,814 points, camera 0, 1280×720,
9,515,389 intersections with compact-box sweep. No point subsampling is needed.

First full-scene raster-only test (40 measured iterations, 20 warmups):

| Binary | Forward + backward raster API median |
|---|---:|
| Rebuilt baseline | 24.745 ms |
| Reduce-scatter | 16.016 ms |

This is a **provisional 1.55× raster-only speedup**, not an end-to-end claim.
Separate-process timing varies significantly on this desktop GPU, so repeated
alternating jobs are required before settling the speedup.

Raster forward RGB and alpha are bit-identical. Relative-L2 screen gradients:
means2d `6.86e-8`, conics `9.50e-7`, colors `6.18e-7`, opacity `2.37e-7`;
all pass the `1e-4` target. JSON evidence currently lives at
`/tmp/opencode/gsplat-blog-{baseline,scatter}-raster.json`.

The first full raw-parameter gradient test flags quaternion relative-L2
`1.57e-4`. Importantly, baseline versus another baseline run itself is
`1.58e-4`, with essentially the same maximum absolute error (~0.033).
This is observed numerical/atomic noise amplified through projection, not
evidence that the strict full-parameter threshold passed. More checks are
needed; **do not call the full-parameter alignment gate passed yet**.
End-to-end timings are also currently noisy and are not accepted as a speedup.

### Repeat measurements and product-wrapper correction

Three independent timing-only jobs (30 warmups, 60 repeats/job) give median of
per-job medians **19.613 → 17.057 ms**, or **1.150× raster-only** for scatter.
This supersedes the initial provisional 1.55× number. Full differentiable
rendering alone did not improve reliably with scatter (60.548 → 61.538 ms).

The product wrapper normalizes raw quaternions before projection. The benchmark
now does that too (in addition to gsplat's internal normalization). New reference
files include that choice in the signature and must not be mixed with old files.

## Experiment 2 — trim fully occluded backward tile suffixes

Implemented a block-wide maximum of existing per-pixel `last_ids` before
backward staging. Unlike the article's full contributor compaction, this removes
only the fully unused tail. It preserves full-width IDs, immutable intersection
lists, and absolute stopping indices, so it supports this >24-bit scene and
channel-chunk reuse without a new forward/backward tensor contract.

| Variant | Raster-only median (3 jobs) | Full differentiable median (3 jobs) |
|---|---:|---:|
| Baseline | 19.613 ms | 60.548 ms |
| Scatter | 17.057 ms | 61.538 ms |
| Scatter + suffix trim | 13.945 ms | 57.061 ms |

Suffix trimming is **1.223× faster than scatter for raster-only**; combined
speedup is **1.406× raster-only / 1.061× full differentiable rendering** in this
initial three-job sequence. These are actual measured numbers, not projections.
Full jobs were grouped by variant and are more sensitive to host/clock drift;
final paired measurements across more cameras remain necessary.

**Correctness:** 112 small regression cases passed against baseline across
channels 1/3/4/8, tile sizes 8/16, optional absgrad, background gradients,
multiple cameras, non-tile-aligned image edges, opacity cutoff/clamp, empty
inputs/intersections and heavy occlusion. Full-scene normalized raw-parameter
alignment also passed: RGB/alpha exact; maximum gradient relative-L2 `8.94e-5`
(quaternions), all below `1e-4`.

Evidence: `benchmarks/blog_eval/results/*-{raster,full}-job*.json`,
`regression-scatter-trim.json`, and `scatter-trim-normalized-alignment.json`.

## Experiment 3 — scalar color projection (rejected)

The article's Step 1 scalar color-gradient accumulation plus invariant
background dot-product hoisting is implemented as a separately saved binary.
It replaces a `COLOR_DIM` running color buffer with its projection onto the
upstream color gradient. Build and all 112 small regression cases passed.
Three full-scene raster jobs measured **14.897 / 13.932 / 14.496 ms** (median
14.496 ms), compared with 13.945 ms for scatter + trim: **no measured win**.
Raw quaternion relative-L2 was `2.03e-4`; the new normalized baseline reference
also exhibited baseline-versus-baseline quaternion noise of `1.99e-4`.
This shows the strict full-parameter gate is unstable for these near-degenerate
rotations, but is not a reason to claim a passing comparison.

**Decision: reverted the scalar accumulator/hoist changes.** Keep the simpler
existing arithmetic because this experiment did not improve the target workload.
Evidence: `regression-scatter-trim-scalar.json`,
`scatter-trim-scalar-normalized-alignment.json`, and the three raster job JSONs.

## Experiment 4 — cooperative selective SH zeroing (rejected)

Adapted Step 3's bulk-zero elimination to this fork's `N*3` color-channel
threads. An in-kernel coalesced sweep wrote zeros only for masked rows and
unused SH coefficients; active coefficients were written once by their VJP.
This avoids the incorrect shortcut of simply returning uninitialized gradients.

All **288 SH regression cases passed** (degrees 0–4, exact and oversized K,
no/all/random masks, N=0/1/31/257, direction gradients enabled/disabled).
But three paired full-render jobs on camera 0 measured median **64.056 ms
without versus 64.973 ms with** selective zeroing. The per-pair times were
60.570/60.418, 66.929/66.780, 64.056/64.973 ms. There is **no reliable gain**.
Division/indexing/predicate cost and mostly masked rows can offset the reduced
zero traffic; this is a hypothesis, not a measured hardware diagnosis.

**Decision: reverted the SH source changes.** This scene's degree-1 SH also
limits the relevance of the article's degree-3 register-streaming optimization.
Evidence: `sh-regression-selective-zero.json`, `*-sh-ablation-job*.json`.

## Experiment 5 — backward alpha predication (provisionally retained)

For channels <=4, misses run the transmittance/color recurrence with alpha=0;
geometry and opacity derivatives are explicitly zeroed unless the pixel is
valid and below the alpha clamp. The warp-wide all-miss skip remains because
gsplat lacks the article's guaranteed-contributor warp mask. Wider channels
retain conditional evaluation. Forward math and termination are unchanged.

All 112 raster regression cases passed. Three paired raster-only jobs measured
median **18.672 → 18.168 ms**, an incremental **1.028×**. This is a small gain
and needs final-binary validation; SH zeroing was present in both binaries but
is irrelevant to raster-only timing. Evidence: `regression-shzero-pred.json`,
`*-pred-ablation-job*.json`.

## Multi-camera check of scatter + suffix trim

The independent timing worker completed 24 serial jobs across camera indices
0/2552/5103, always using every point, with 30 warmups and 80 timed iterations.
Paired raster-only speedups were **1.370× / 1.476× / 1.360×**. Full-render
speedups were **1.099× / 1.053× / 0.938×**, respectively: one view regressed.
This rules out claiming a universal end-to-end speedup from the first camera.
See `benchmarks/blog_eval/results/PAIRED_SUMMARY.md` and its raw JSON files.

## Experiment 6 — forward warp sub-tile culling (rejected)

Implemented a specialized tile16 narrow-channel forward path with 8×4 warp
pixel ownership, per-staged-splat conservative opacity-aware ellipse AABBs,
and ballot/ffs worklists. Unlike the article this is an AABB mask, not an
exact conic row-span mask. Generic tile/channel paths are retained.
The first FP64-bound version passed all 112 regression cases but measured only
**18.227 → 18.027 ms** raster-only (~1%): too small/noisy to accept the extra
complexity. The FP32-bound version also passed 112 cases, but three paired jobs
measured only **16.525 → 16.195 ms** raster-only (1.020×). Forward-only medians
were **4.796 → 4.572 ms** (1.049×), with considerable job-to-job noise.
The AABB is looser than the article's exact conic mask and many ill-conditioned
conics take the conservative fallback. These are implementation differences,
not evidence that the article's exact-mask technique is ineffective.

**Decision: revert the forward-culling prototype.** Its extra specialized loop,
fallbacks and heuristic rounding margins are not justified by these small/noisy
gains. The independent audit also identified an overflowing-float baseline
compatibility risk in the earlier FP64-bounds version. No final forward source
change will be shipped from this experiment.

## Independent double-check checkpoint

A separate code audit confirmed the reduce-scatter ownership and normal-range
suffix trimming, but found important gaps:

- Alpha=0 alone is insufficient for branchless gradients: `0*Inf` can introduce
  NaNs on rejected lanes even with finite but overflowing conics. Predication is
  being hardened and must pass new adversarial cases before acceptance.
- Empty trailing tile endpoints near `INT_MAX` need a uniform empty-range exit
  before forming the exclusive trimmed endpoint.
- The original 112-case matrix does **not** cover actual N=0, packed mode,
  multi-batch tile16 lists, tile masks or non-default streams. Its earlier
  "empty inputs" description was too broad: it covered empty intersections.
  Targeted coverage is being added.
- Baseline SH backward's default-stream launch and masked-tile uninitialized
  alpha are inherited issues. Rejected SH/culling experiments do not fix them;
  they will be recorded separately rather than silently bundled into this
  optimization work.
- Results need immutable binary/source identity. The timing harness now hashes
  binaries; regression identities and a build manifest are being added.

Static SM89 RGB-backward SASS inspection supports the intended reduction change:
**55 → 13 SHFL instructions, 11 → 1 RED instructions, 40 → 56 registers/thread**.
These are static compiler outputs, not runtime instruction counts or speedups.

## Hardened integration checkpoint

The final candidate is gsplat commit **`b96efa8`** on
`perf/blog-cuda-speedups`, following `44b9b41`. Only the backward rasterizer and
the new reduce-scatter header differ from the original baseline; forward and SH
sources are restored. The final changes are contained in this gsplat checkout.

Rejected lanes now have genuinely zero contributions: invalid lanes do not
update the color buffer, and invalid/clamped lanes skip geometry/opacity/absgrad
arithmetic entirely. This addresses the audit's `0*Inf` contamination example.
Empty tile ranges return uniformly; the exclusive trimmed endpoint is widened
before narrowing. These fixes are retained, not just documented.

**126 expanded differential cases passed**, including actual zero inputs,
multi-batch tile16 lists, packed camera layouts, channel-list reuse, sparse tiles,
finite overflowing conics and rejected nonfinite upstream lanes. The same matrix
passed **CUDA compute-sanitizer memcheck with 0 errors** (exit 0).
Tile-mask and non-default-stream baseline issues remain explicitly excluded.

Final camera-0 raster ablation, with no compiler running (3 jobs, 30 warmups,
100 iterations/job):

| Variant | Median of job medians |
|---|---:|
| Baseline | 22.046 ms |
| Scatter + trim | 17.321 ms |
| Scatter + trim + safe predication | 16.692 ms |

This confirms a **1.321× combined raster-only speedup** in this sequence and
an incremental **1.038×** from safe predication. Absolute timings differ from
earlier desktop runs; do not combine baselines from different sequences.
The final RGB backward kernel uses **48 registers and no stack/local memory**
according to SM89 resource output. Binary/source identities and build flags are
recorded in `benchmarks/blog_eval/results/build_manifest.json`.

This checkpoint preceded the final paired/alignment runs; the results at the
top of this report and the following sections supersede its pending status.

## Final-binary alignment recheck

The final commit/binary—not an earlier experimental variant—was compared with
a fresh rebuilt-baseline reference on every point, camera 0:

| Quantity | RGB relative-L2 | RGB+ED / absgrad relative-L2 |
|---|---:|---:|
| Output image / alpha | 0 / 0 (bit-identical) | 0 / 0 (bit-identical) |
| World means | 1.64e-6 | 3.33e-6 |
| Raw quaternions | **1.11e-4** | **1.63e-4** |
| Log scales | 6.59e-6 | 5.22e-6 |
| Opacity logits | 2.93e-7 | 2.03e-7 |
| SH coefficients | 7.35e-7 | 6.99e-7 |
| Absolute screen means | Disabled | 2.45e-7 |

Both strict full-parameter JSON results are intentionally **`passed: false`**
because raw quaternion relative-L2 exceeds `1e-4`; the threshold was not raised
to turn them green. Candidate-versus-candidate quaternion noise is `1.14e-4`
(RGB) / `1.35e-4` (RGB+ED); corresponding baseline repeat noise in these pairs
is `7.15e-5` / `4.60e-5`, with earlier baseline repeats up to ~`2e-4`.
This is consistent with atomic accumulation noise amplified by projection of
near-degenerate splats, but no statistical proof of training equivalence follows.
There was no training-convergence run.

The **direct full-scene RGB raster check passes**: bit-identical RGB/alpha,
means2d `1.00e-7`, conics `2.14e-6`, colors `6.18e-7`, opacity `1.90e-7`
relative-L2. Large conic-gradient magnitudes mean maximum absolute deltas can
be much larger than the relative norm; both are retained in raw JSON evidence.

The direct raster regression also passes **all 126 cases on a non-default CUDA
stream**, including input production and gradient consumption on that stream.
This does not clear the inherited SH-stream issue in the full rendering path.

## Article elements not ported in this evaluation

- **Two-splat folding:** single-splat halving reduction is retained; folding
  two consecutive splats would require a wider live state and more traversal
  changes. The measured single-splat win is not a measurement of folding.
- **Two-stage sorting:** promising next investigation, but not implemented or
  timed independently here. This view has only 3,600 tiles (16-bit tile IDs fit),
  whereas the general API supports larger grids and multiple cameras. A port
  needs stable equal-depth ordering and safe key-width fallbacks.
- **Packed 48-byte splats / exact conic sub-tile masks:** the tested culler used
  conservative AABBs, not the article's exact conic mask or packed attribute
  structure. Rejecting that prototype is not a negative result for the exact
  published design.
- **Full contributor compaction:** not implemented. The supplied scene exceeds
  the article's 24-bit Gaussian-ID representation; a separate mask or remapping
  could avoid that limit. The retained suffix trim is a smaller compatible
  optimization, not equivalent to dropping every interior non-contributor.
- **Multirow execution / pipe balancing:** not implemented. Ada's FP32 and
  residency characteristics differ from SM80; importing A100 register caps or
  launch bounds without profiling could regress this GPU. More pixels/thread
  also increases live state for generic depth/feature channels.
- **Base-2 exponential:** existing `__expf` with `--use_fast_math` already uses
  the hardware fast exponential path. Merely renaming it to `exp2` is not a new
  speedup. Coefficient rescaling/explicit quadratic rounding would require their
  own cutoff-boundary validation; they were not shipped.
- **BF16 SH:** not implemented or quality-tested. This degree-1 checkpoint has
  one quarter of the degree-3 SH payload motivating the article's tables, and
  quantizing coefficients would change the numerical/optimizer contract.

The article's headline compares different implementations/scenes on an A100.
This work measures incremental changes against this particular fork on a 4090.
Neither its headline speedup nor the unimplemented steps' gains are claimed.

## Production RGB+ED / absgrad and final verification

`results/PRODUCTION_SUMMARY.md` records three paired full-scene jobs using the
wrapper's default four-channel/absolute-gradient settings. The median paired
speedups are **1.346× raster-only and 1.081× full rendering**. All three pairs
favor the candidate, but absolute latency varies substantially; this is one
camera, not evidence of a uniform all-camera or whole-training gain.

The independent production direct-raster reference check passes with exact
image/alpha and relative-L2: means2d `1.30e-7`, conics `1.70e-6`, colors
`8.40e-7`, opacity `2.29e-7`, absgrad `2.56e-7`. Peak allocated memory is unchanged
in these jobs (4.279 GiB raster-only / 6.915 GiB full). The full raw-quaternion
limitation above remains; timing finite checks are not substituted for alignment.

Final verification evidence:

- 126/126 adversarial raster cases pass on the default stream and 126/126 on a
  non-default stream.
- CUDA **memcheck and synccheck both report 0 errors**, using the final binary
  and expanded matrix. Racecheck and an exhaustive numerical proof were not run.
- `black --check`, Python compilation, and `git diff --check` pass.
- Final source hashes, commit `b96efa8`, tested binary and worktree `csrc.so`
  agree with `results/build_manifest.json`. Candidate SHA-256:
  `22031d5217692b6c0dc01fcc871d7f43b3bb24fc6ea2d73bc0b200504d4a307d`.
- The gsplat worktree is clean; GLM stays at pinned `33b4a621…` with no source
  changes. Only backward rasterization and its helper differ from `ffe2db3`.
- This gsplat checkout was built with `setup.py build_ext --inplace`; no package-install command was used.

**Integration:** gsplat branch `perf/blog-cuda-speedups` contains commits
`44b9b41`, `b96efa8` and the partial-warp correction. The final tutorial and
evaluation artifacts are stored in this repository beside the implementation.
This is an evidence-backed optimization/review branch, **not a claim of verified
training convergence or universal production acceleration**.

## Supplementary averaged-gradient check

To distinguish repeat accumulation noise from a systematic gradient difference,
an additional check uses a **fixed 16 renders/backwards per binary and mode**,
identical parameters, camera and upstream gradients, and float64 CPU running
means. The repeat count was chosen before collecting these data; it is not
adaptively increased until a pass. The gradient threshold remains `1e-4` and
forward maximum-absolute threshold remains `2e-5`.

This estimates agreement of average gradients. It does **not** retroactively
make the single-pass strict quaternion diagnostics pass, remove per-step noise,
or demonstrate convergence.

**Both fixed-16 comparisons pass on rebuilt `88513ea`.** Mean image/alpha are
exact. Mean-gradient relative-L2:

| Gradient | RGB | RGB+ED + absgrad |
|---|---:|---:|
| Means | 6.24e-7 | 7.36e-7 |
| Raw quaternions | 4.57e-5 | 3.50e-5 |
| Log scales | 1.48e-6 | 2.00e-6 |
| Opacity logits | 1.77e-7 | 9.55e-8 |
| SH coefficients | 2.53e-7 | 3.13e-7 |
| Absolute screen means | Disabled | 1.55e-7 |

All 16 passes contribute; no retries, discarded samples or threshold changes.
The baseline-only script initially had compact-box disabled; before collecting
candidate data it was corrected to compact-box **true / sweep**, and both
baseline references were regenerated with that setting. JSON records the
correction. Mean agreement is useful additional evidence, not a confidence
interval or proof of per-step/training equivalence.

## Partial-warp regression fix and rebuilt safety gates

A direct 4×4-tile test exposed nonzero baseline opacity gradients (~17.697)
becoming zero in `b96efa8`. That tile launches 16 threads: the halving helper's
32-lane component ownership cannot be used with missing lanes.

`88513ea` routes **every block with `block_size % 32 != 0`** through the original
component-wise reduction and lane-0 atomics, including partial final warps such
as a 6×6 tile. Complete-warp narrow-channel blocks retain reduce-scatter; wide
channels retain the baseline path. The helper itself is not made to pretend
that an active mask fixes missing component owners.

The expanded matrix now has **166 cases**, including tile4 with channels
1/2/3/4/8 and absgrad on/off, plus tile6 cases. All pass against a fresh baseline
reference on both default and non-default streams. The rebuilt binary also
passes both memcheck and synccheck over all 166 cases with **0 errors**.
Evidence: `results/partial-warp-{baseline,final-default,final-nondefault}.json`
and `partial-warp-final-{memcheck,synccheck}.{json,log}`.

Rebuilt source/binary identity:

- gsplat commit `88513ea72203b1d8f0ebb7e6e27d7ce024574702`.
- Binary `final-partial-warp-safe.so`, SHA-256
  `6160354698af48c34a5e6a090c5e241c564959fb6e0d54d2c55bb3d9c8373ddb`.
- Worktree `csrc.so` matches that binary; source hashes match the manifest.
- RGB backward still uses 48 registers, no stack/local memory. The compiled
  kernel now also contains the baseline fallback, so old scatter-only static
  instruction counts must not be presented as total counts for this binary.

## Rebuilt delivery sign-off (`88513ea`)

`results/REBUILT_FINAL_SUMMARY.md` and its **36 new paired timing JSONs** now
supersede the earlier b96 timing evidence; none of the old timings are pooled
into the executive-summary numbers. All loaded binary hashes match the rebuilt
manifest. No compilation or other task-owned GPU workload overlapped these runs.

- RGB raster gains: **1.337× / 1.648× / 1.363×** on cameras 0/2552/5103.
  Full-render ratios: **1.100× / 1.016× / 1.039×**. Camera 2552's full-path
  pairs change direction, so its small aggregate gain is inconclusive.
- RGB+ED + absgrad, camera 0: **1.311× raster / 1.081× full**, reproduced in
  all three pairs for each mode.
- Rebuilt direct-raster alignment: exact image/alpha, worst gradient relative-L2
  **2.62e-6 RGB / 1.48e-6 RGB+ED**, both below unchanged `1e-4`.
- Both rebuilt full-path single-shot checks again fail **only raw quaternions**
  (~`1.63e-4`). They were run once each, not retried. Within-candidate quaternion
  repeat noise is `2.22e-4` / `1.40e-4`; this is context, not a threshold waiver.
- The separate fixed-16 mean-gradient checks pass all parameter groups. Mean
  quaternion relative-L2 is **4.57e-5 / 3.50e-5**, without threshold relaxation.
- Final full-scene intersection IDs reach **20,789,812**, directly exercising
  IDs beyond 24 bits; no truncating contributor-mask packing was introduced.

The source-to-binary checks, **166-case default/non-default suites**, both
sanitizers, ensemble reports, paired timing identities and explicit single-shot
limitations are consolidated in `results/final_verification.json` and
`results/build_manifest.json`. Python compilation, Black and diff checks pass.
The gsplat worktree is clean at the tested revision, and the nested GLM dependency source/pin remains unchanged.

**Recommendation:** keep this as the tested optimization/review branch. It
delivers reproducible raster gains on the supplied workload, with compatible
partial-warp fallbacks. Do not describe it as a universal whole-training gain,
strict single-shot quaternion-aligned implementation, or convergence-validated
production release. Those would require additional evidence.

### SH streaming applicability (source assessment)

The article's starting point keeps 48 forward coefficients plus 48 coefficient
gradients live in one thread. This fork instead runs SH backward across `N*3`
threads (one color channel per thread), with direct coefficient-gradient writes
in `sh_coeffs_to_color_fast_vjp`. Consequently the article's register-window
transformation is not a direct fix for the same register problem here. Moreover,
the supplied PLY has degree-1 rather than degree-3 SH. SH zero-fill removal may
still help, but must write zeros for masked points and unused higher-degree
coefficients; it is not valid to replace `zeros_like` with `empty_like` alone.

### Full-scene contributor packing applicability

The article packs contributor flags in the top 8 bits of a 32-bit Gaussian ID,
leaving 24 bits. This scene contains 20,789,814 points (>16,777,216). A direct
24-bit adaptation would corrupt IDs. A safe implementation needs a separate
mask buffer, wider storage, or an explicit remapping/fallback.
