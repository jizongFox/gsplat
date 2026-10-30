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
- no contribution mask (`cmask`) was added, so this isolates synchronization
  and staging from mask-generation costs.

The candidate keeps the original recurrence, alpha/clamp/nonfinite guards,
warp reductions, and global gradient atomics. The baseline and candidate were
freshly built from the same worktree and CUDA/PyTorch environment.

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

The first implementation is **not retained or pushed**. It is useful evidence
that warp-private staging can be made race-free and numerically compatible, but
the measured gain is workload-dependent and the 4K aggregate regresses. A
second iteration could reduce the 64-register cost or add a separately measured
contribution mask, but that would be a new experiment rather than evidence for
this candidate.

Reproduction artifacts are local under `build/warp-private/`; the candidate is
not part of PR11 and no binary is committed.
