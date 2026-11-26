# PennyGS (arXiv 2611.27525) Sec. 3.4: forward subtile masks and per-warp hit lists

Part of the PennyGS series on gsplat: each contribution is a separate opt-in change
measured against `dev/dc` at `29ee2a0` in one process on one build, variants
alternating inside each repeat (20 warmups, 80 repeats, medians).

Workload: RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8; water2 3DGS model,
2,909,851 DC-colour Gaussians; 1280x720 views 0, 250, 500, 750, 1000, 1250 of its
`cameras.json` (external inputs, not committed); the parent application's
renderer shape (RGB + camera normal + expected depth, 7 channels padded to 8,
absgrad, compact-box sweep, tile 16).

## Change

`rasterize_to_pixels(..., subtile_masks=True)` /
`rasterization(..., rasterize_subtile_masks=True)`, tile size 16:

- the 16x16 tile is split into eight 8x4 subtiles, one per warp (the default maps
  a warp to two 16-pixel rows);
- when a batch of Gaussians is staged, the loading thread also records which
  subtiles the Gaussian's alpha >= 1/255 ellipse can reach: a bounding-box test of
  `0.5 d^T Q d <= ln(255 opacity) + err` (VkSplat's subtile masks). `err` bounds the
  float32 rounding of the renderer's sigma over the tile (2^-20 times the largest
  sum of the quadratic terms at the tile corners), because cancellation can make
  the renderer accept pixels the exact ellipse excludes; ill-conditioned conics
  (det below 1e-4 q00 q11), degenerate or nonfinite inputs reach every subtile;
- each warp compacts the batch entries that reach its subtile into a hit list,
  in depth order, with ballots, and blends only those (HiGS's per-warp hit lists).

Skipped entries have alpha < 1/255 at every pixel of the subtile, which the
default kernel skips pixel by pixel, so images, alphas and `last_ids` are
identical. The backward kernel is unchanged.

Colour staging, the other half of PennyGS's forward change, is added to the
subtile kernel for 1-4 channels only: see `subtile_staging.md`. It does not
apply to the 7-channel (padded to 8) renderer measured below, whose subtile
kernel is the one measured here.

Resources (8 channels): 48 registers and 11.5 KB static shared memory per
256-thread block, against 40 registers and 7 KB for the default.

## Results

`subtile_benchmark.py --scope raster` (forward of `rasterize_to_pixels`, ms):

| View | Intersections | default | subtile | Speedup | Outputs |
|---|---:|---:|---:|---:|---|
| 0 | 1,011,765 | 0.814 | 0.711 | 1.146x | identical |
| 250 | 4,830,907 | 4.153 | 3.047 | 1.363x | identical |
| 500 | 996,736 | 0.789 | 0.682 | 1.157x | identical |
| 750 | 1,781,205 | 1.198 | 0.990 | 1.211x | identical |
| 1000 | 754,760 | 0.761 | 0.684 | 1.112x | identical |
| 1250 | 619,808 | 0.650 | 0.585 | 1.112x | identical |
| **geomean** | | | | **1.180x** | |

`--scope render` (full renderer forward + backward, ms):

| View | 0 | 250 | 500 | 750 | 1000 | 1250 |
|---|---|---|---|---|---|---|
| default | 16.25 | 26.47 | 16.37 | 17.93 | 16.05 | 15.68 |
| subtile | 16.12 | 25.39 | 16.26 | 17.71 | 15.97 | 15.60 |
| speedup | 1.008x | 1.043x | 1.007x | 1.012x | 1.005x | 1.005x |

Geomean **1.013x**; images identical; gradients within the run-to-run noise of
the backward atomics (max rel-L2 5.9e-05).

Measurements: `measurements/subtile_{raster,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_subtile_masks.py
PYTHONPATH=$PWD python benchmarks/pennygs/subtile_benchmark.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --scope raster --output raster.json   # then --scope render
```
