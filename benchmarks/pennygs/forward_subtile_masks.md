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
  `0.5 d^T Q d <= ln(255 opacity)` with a margin; degenerate or nonfinite inputs
  reach every subtile (VkSplat's subtile masks);
- each warp compacts the batch entries that reach its subtile into a hit list,
  in depth order, with ballots, and blends only those (HiGS's per-warp hit lists).

Skipped entries have alpha < 1/255 at every pixel of the subtile, which the
default kernel skips pixel by pixel, so images, alphas and `last_ids` are
identical. The backward kernel is unchanged. Colour staging, the other half of
PennyGS's forward change, is the separate open PR #11.

Resources (8 channels): 48 registers and 11.5 KB static shared memory per
256-thread block, against 40 registers and 7 KB for the default.

## Results

`subtile_benchmark.py --scope raster` (forward of `rasterize_to_pixels`, ms):

| View | Intersections | default | subtile | Speedup | Outputs |
|---|---:|---:|---:|---:|---|
| 0 | 1,011,765 | 0.815 | 0.713 | 1.143x | identical |
| 250 | 4,830,907 | 4.128 | 3.065 | 1.347x | identical |
| 500 | 996,736 | 0.788 | 0.683 | 1.154x | identical |
| 750 | 1,781,205 | 1.192 | 0.995 | 1.198x | identical |
| 1000 | 754,760 | 0.757 | 0.685 | 1.105x | identical |
| 1250 | 619,808 | 0.649 | 0.585 | 1.111x | identical |
| **geomean** | | | | **1.174x** | |

`--scope render` (full renderer forward + backward, ms):

| View | 0 | 250 | 500 | 750 | 1000 | 1250 |
|---|---|---|---|---|---|---|
| default | 16.22 | 26.49 | 16.44 | 17.99 | 16.10 | 15.74 |
| subtile | 16.14 | 25.41 | 16.33 | 17.71 | 16.03 | 15.65 |
| speedup | 1.005x | 1.042x | 1.007x | 1.015x | 1.004x | 1.006x |

Geomean **1.013x**; images identical; gradients within the run-to-run noise of
the backward atomics (max rel-L2 6.3e-05).

Measurements: `measurements/subtile_{raster,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_subtile_masks.py
PYTHONPATH=$PWD python benchmarks/pennygs/subtile_benchmark.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --scope raster --output raster.json   # then --scope render
```
