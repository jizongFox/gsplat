# Independent warp-private backward traversal experiment

## Scope

This experiment starts from the fresh `dev/dc` merge commit `a7050c6` in a
separate worktree. It is independent of PR11 and does not modify the parent
Gaussian-splatting checkout, the original gsplat checkout, or PR11.

The tested idea is the blog's barrier-free backward traversal, adapted to
gsplat's 16×16 tile:

- one CUDA block still owns one tile and has eight 32-thread warps;
- each warp gets a private 32-entry slice of the backward staging arrays;
- each warp uses its own maximum `last_ids` endpoint;
- full-warp cases use `__syncwarp()` before reuse and after loading a batch;
- partial physical warps and unsupported/wide paths retain the baseline
  block-wide synchronization path;
- a follow-up tested a conservative on-the-fly warp contribution test for cmask;
  it does not pack metadata into `flatten_ids`. The cmask code is not retained.

The candidate keeps the original recurrence, alpha/clamp/nonfinite guards,
warp reductions, and global gradient atomics. The baseline and candidate were
freshly built from the same worktree and CUDA/PyTorch environment.

## Code changes in this PR

- `gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu`: add a compile-time
  `WARP_PRIVATE` kernel specialization. Full physical-warp blocks allocate
  eight independent 32-entry staging slices, compute a per-warp absolute
  `last_ids` endpoint, and use `__syncwarp()` around each slice's load/reuse.
  The existing path remains selected for partial physical warps and for
  `COLOR_DIM > 8`; the launch dispatch selects the private path only when
  `tile_size * tile_size` is divisible by 32. Global gradient atomics remain
  unchanged because different warps can still update the same Gaussian.
- `benchmarks/blog_eval/gsplat_raster_regression.py`: append deterministic
  imbalance, staging-boundary, logical-7/physical-8, ordinary, and partial-warp
  fixtures without changing PR9 fixture order or seeds.
- `benchmarks/warp_private/benchmark_seven_channel.py`: add a standalone,
  content-addressed COLMAP/PLY raster harness for RGB + camera-Z depth +
  camera-space Gaussian local +Z normal. Timing excludes projection, SH and
  feature construction; references and all five gradient outputs use the PR9
  gates.
- `tests/test_warp_private_benchmark.py` and
  `tests/test_warp_private_layout.py`: validate the harness, normal convention,
  reference identity and fixture coverage.

## Production renderer context and cmask follow-up

Inspection of `dc_reality/splatting/renderers/gsplat_renderer` shows two real
layouts. `gsplat_renderer_3d_normals.py` sends RGB + per-Gaussian normal and
requests `RGB+ED`, resulting in seven logical raster channels. `pgsr_renderer.py`
sends RGB + camera depth + normal + plane depth, i.e. eight channels, with
`packed=False`, `absgrad=True`, pinhole cameras and compact-box sweep. The
standalone harness was extended to model the latter eight-channel feature
layout, including minor-axis Gaussian normals and plane distance.

The cmask follow-up uses PR13 warp-private as its baseline and computes a
conservative ellipse-vs-warp-subrectangle test on the staged Gaussian. It skips
only when the whole warp rectangle is outside the alpha-visible ellipse; it
does not alter `flatten_ids`, and invalid conics conservatively remain active.
This is deliberately a first isolation experiment, not the final metadata-buffer
design from the blog.

| PGSR-style 8-channel raster workload | Forward | Backward | Combined |
|---|---:|---:|---:|
| Native view0 | 1.008× | 0.915× | 0.935× |
| Native view623 | 1.006× | 0.976× | 0.982× |
| Native view1247 | 0.991× | 0.967× | 0.962× |
| Native geometric mean | **1.002×** | **0.952×** | **0.959×** |
| 4K geometric mean | **0.944×** | **0.934×** | **0.943×** |

All six production-style parity comparisons pass; maximum gradient relative-L2
is below `5e-7`. However, the cmask path increases the `COLOR_DIM=8` private
kernel from 64 to **72 registers/thread** and is slower in both native and 4K
combined timings. It also fails four guard-focused cases in the expanded
193-case matrix: `target_finite_overflow_abs{0,1}` and
`target_rejected_nonfinite_upstream_abs{0,1}` (color and opacity gradients).
This is a correctness blocker, so the cmask CUDA specialization is excluded
from the published implementation. The production timing remains evidence
only; no cmask speedup is claimed.

## Real workload

The primary workload is the requested **RGB + depth + normal** path:

- RGB: 3 channels;
- camera-Z depth: 1 channel;
- camera-space Gaussian local +Z normal: 3 channels;
- seven logical channels, padded by the public wrapper to CUDA `COLOR_DIM=8`.

The supplied water2 scene uses all 1,932,132 Gaussians. The raster benchmark
uses COLMAP-derived cameras, tile16, float32, compact-box sweep, absgrad, and
detached raster timing. Three jobs per binary alternate base/candidate order;
each job uses 40 warmups and 80 timed samples. The table uses the median of job
means and `baseline_ms / candidate_ms`.

## Results

### Native 1280×720, six selected views

| View | Forward | Backward | Forward + backward |
|---:|---:|---:|---:|
| 0 | 1.003× | 1.000× | 1.002× |
| 623 | 1.001× | 0.953× | 0.959× |
| 1247 | 0.994× | **1.258×** | **1.169×** |
| 156 | 1.002× | 1.012× | 1.011× |
| 468 | 0.997× | 1.028× | 1.027× |
| 936 | 0.999× | 1.030× | 1.022× |
| **Geometric mean** | **0.999×** | **1.062×** | **1.030×** |

The view1247 result is unusually favorable, while view623 regresses. The
six-view aggregate is only about 3.0% and is not stable enough to claim a
general improvement from this one scene.

### 4K, three selected views

| View | Forward | Backward | Forward + backward |
|---:|---:|---:|---:|
| 0 | 1.000× | 1.005× | 1.004× |
| 623 | 1.000× | 0.983× | 0.984× |
| 1247 | 0.981× | 0.980× | 0.979× |
| **Geometric mean** | **0.994×** | **0.989×** | **0.989×** |

At 4K the candidate is about 1.1% slower overall. The result does not support
retaining this implementation as a broad speedup.

These timings are directional evidence, not a claim that the blog's A100
14.9%-barrier result transfers to this Ada workload. The candidate's warp-
private `COLOR_DIM=8` kernel allocates 64 registers per thread versus 56 for
the baseline; that resource increase is a likely reason the synchronization
savings do not consistently win.

## Why the speedup is not broad

The experiment changes one synchronization mechanism, but it does not remove
the main backward work. Every surviving Gaussian still requires the same
recurrence through transmittance and the color buffer, the same warp reduction,
and the same global gradient atomics. A warp-private slice removes cross-warp
waiting only when warp workloads are sufficiently imbalanced; it does not make
those arithmetic, reduction, or atomic operations cheaper.

There is also a resource trade-off. On the actual 8-channel path, the private
kernel increases allocation from **56 to 64 registers/thread**. The extra live
slice pointers and private traversal state can reduce resident-warps/SM and
therefore reduce the latency hiding available for global loads and atomics.
This can cancel the barrier saving. The measured 4K result (0.989× combined)
is consistent with that trade-off; it is evidence, not a profiler-counter
claim of exact occupancy.

The result is workload-dependent because the useful barrier work is workload-
dependent. View1247 reaches **1.169×** native combined, while view623 reaches
**0.959×**. At 4K, the three-view geometric mean is **0.989×**. A blog result
that attributes 14.9% of cycles to barriers on another GPU/kernel mapping
cannot be transferred directly to this Ada/gsplat mapping.

The cmask follow-up makes the same trade-off visible. It can skip a warp only
when a conservative alpha-visible ellipse test proves that the warp rectangle
cannot contribute. Many Gaussians remain possible contributors, so the test
cost is paid without a skip. The first implementation also adds live state:
the 8-channel private kernel rises from **64 to 72 registers/thread**. In the
PGSR-style production layout its measured ratios were **0.959× native
combined** and **0.943× 4K combined**, before considering its correctness
failure on four guard-focused stress cases. Thus the cmask code is not included
in the implementation. The failures involved overflow/nonfinite guard cases
and color/opacity gradients, showing that “skip if the ellipse misses” must be
integrated with the existing identity-transmittance and nonfinite semantics,
not added as an unqualified branch.

In short:

```text
less block waiting
  - extra registers / lower latency hiding
  - unchanged recurrence, reductions and atomics
  - cmask test/branch overhead and conservative false positives
  = no broad net speedup on this workload
```

## Correctness

- Baseline and candidate each pass **193 differential raster fixtures**.
- The original PR9 fixture order/seeds remain preserved; appended cases cover
  warp imbalance, staging boundaries, logical 7/physical 8 channels, ordinary
  paths, and partial-warp fallback.
- Candidate 7-channel scene parity passes unchanged image/alpha and gradient
  gates; all five gradient outputs are compared.
- Compute Sanitizer: memcheck **0 errors**, synccheck **0 errors**, racecheck
  **0 hazards / 0 errors / 0 warnings** over the 193-case matrix.
- **117 CPU tests pass** in this experimental worktree.

## Decision

The first implementation is **not a general speedup claim**. This PR records
that warp-private staging can be made race-free and numerically compatible, but
the measured gain is workload-dependent and the 4K aggregate regresses. A
second iteration could reduce the 64-register cost or add a separately measured
contribution mask, but that would be a new experiment rather than evidence for
this candidate. This PR intentionally publishes the experiment and its
negative/mixed result for review; it does not claim that the candidate should
replace the default path.

Reproduction artifacts are local under `build/warp-private/`; no binary is
committed. This is a new experiment PR, separate from PR11.
