# Compact Box (CB) for Tile Intersection

This repository includes a FastGS-style Compact Box (CB) pruning path in tile
intersection for `gsplat`.

## What it does

- CB prunes Gaussian-tile pairs in `isect_tiles` using Mahalanobis distance in
  projected 2D space.
- The pruning happens before sorting and rasterization, so it reduces
  `n_isects` directly.
- Default behavior is unchanged: CB is opt-in.

## Where it is implemented

- CUDA tile intersection kernel: `gsplat/cuda/csrc/isect_tiles.cu`
- Python tile intersection wrapper: `gsplat/cuda/_wrapper.py`
- Python rasterization API entrypoint: `gsplat/rendering.py`

## API

`gsplat.rendering.rasterization()` now exposes:

- `compact_box: bool = False`
- `compact_box_mult: float = 1.0`
- `compact_box_tau2: Optional[float] = None`

These are passed through to `isect_tiles()`.

## Threshold behavior

- If `compact_box=False`, no CB pruning is applied.
- If `compact_box=True` and `compact_box_tau2` is provided, that value is used
  directly as the squared Mahalanobis threshold.
- If `compact_box=True` and `compact_box_tau2` is `None`, a provisional mapping
  is used:

  `tau2 = 9.0 * compact_box_mult^2`

This mapping is intentionally simple for PoC and can be replaced later with an
exact FastGS-equivalent formula if needed.

## Tile predicate

For each candidate tile from the existing radius-based coarse box, the kernel
computes

`min_{(x, y) in tile_rect} [x, y] Q [x, y]^T`

where `Q` is the projected conic (`cov2d^{-1}`) and `(x, y)` is relative to the
Gaussian center.

The pair is kept iff:

`min_d2 <= tau2`

The rectangle uses a continuous pixel-center convention:

- `x in [tile_x * tile_size + 0.5, (tile_x + 1) * tile_size - 0.5]`
- `y in [tile_y * tile_size + 0.5, (tile_y + 1) * tile_size - 0.5]`

## Quick usage

```python
from gsplat.rendering import rasterization

render_colors, render_alphas, meta = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    packed=False,
    compact_box=True,
    compact_box_mult=1.0,
    # compact_box_tau2=9.0,  # optional explicit threshold
)

print(meta["tiles_per_gauss"].float().mean(), meta["isect_ids"].numel())
```

## A/B benchmark script

Use `examples/benchmark_compact_box.py` to compare baseline and CB on a
synthetic scene:

```bash
PYTHONPATH=. python examples/benchmark_compact_box.py \
  --packed False \
  --iters 20 \
  --warmup 5 \
  --compact-box-mult 1.0
```

The script reports:

- `n_isects`
- mean `tiles_per_gauss`
- average render time
- image difference (`max_abs`, `mean_abs`, `rmse`, `psnr`) between baseline and CB
