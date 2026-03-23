# Compact Box (CB) for 2DGS

This note describes Compact Box tile pruning for `rasterization_2dgs()`.

## What it does

- CB prunes Gaussian-tile pairs in `isect_tiles` before sorting/rasterization.
- For 2DGS, the pruning conic is derived from `ray_transforms` (homography-aware
  geometry) instead of 3DGS `conics`.
- Default behavior is unchanged: CB is opt-in.

## API

`gsplat.rendering.rasterization_2dgs()` exposes:

- `compact_box: bool = False`
- `compact_box_mult: float = 1.0`
- `compact_box_tau2: Optional[float] = None`
- `compact_box_impl: Literal["rect_min", "sweep"] = "sweep"`

## Threshold behavior

- If `compact_box=False`, no CB pruning is applied.
- If `compact_box=True` and `compact_box_tau2` is provided, that value is used
  directly.
- If `compact_box=True` and `compact_box_tau2` is `None`, per-Gaussian threshold
  is used:

  `tau2_i = compact_box_mult * 2 * log(opacity_i * 255)`

- If `tau2_i <= 0`, the Gaussian is skipped in CB mode.

## Implementation modes

- `rect_min`: per-tile rectangle minimum test (reference path)
- `sweep`: slice/span traversal (faster path)

## 2DGS-specific robustness

- CB derives a 2D conic approximation from each Gaussian's `ray_transforms`.
- If conic derivation is ill-conditioned, CB falls back to baseline coarse
  radius-box traversal for that Gaussian (no aggressive hard skip).

## Quick usage

```python
from gsplat.rendering import rasterization_2dgs

render_colors, render_alphas, render_normals, surf_normals, render_distort, render_median, meta = rasterization_2dgs(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    compact_box=True,
    compact_box_mult=1.0,
    compact_box_impl="sweep",
)

print(meta["tiles_per_gauss"].float().mean(), meta["isect_ids"].numel())
```

## Benchmark script

Use `examples/benchmark_compact_box_2dgs.py`:

```bash
PYTHONPATH=. python examples/benchmark_compact_box_2dgs.py \
  --iters 20 \
  --warmup 5 \
  --compact-box-mult 1.0
```

Sweep multipliers in one run:

```bash
PYTHONPATH=. python examples/benchmark_compact_box_2dgs.py \
  --sweep-mults 0.7 1.0 1.3
```

## Recommended profile

- Default/safe profile: `compact_box=True`, `compact_box_impl="sweep"`,
  `compact_box_mult=1.0`.
- If you need more speed and can accept a slightly larger image delta,
  try `compact_box_mult=0.8`.
- If you need stricter visual parity with baseline, use `compact_box_mult=1.2`
  or set explicit `compact_box_tau2`.
