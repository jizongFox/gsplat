# PennyGS (arXiv 2611.27525) Sec. 3.3: single-pass key emission

Part of the PennyGS series on gsplat: each contribution is a separate opt-in change
measured against `dev/dc` at `29ee2a0` in one process on one build, variants
alternating inside each repeat (20 warmups, 80 repeats, medians).

Workload: RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8; water2 3DGS model,
2,909,851 DC-colour Gaussians; 1280x720 views 0, 250, 500, 750, 1000, 1250 of its
`cameras.json` (external inputs, not committed); the parent application's
renderer shape (RGB + camera normal + expected depth, 7 channels padded to 8,
absgrad, compact-box sweep, tile 16).

## Change

`isect_tiles(..., single_pass_capacity=K)` / `rasterization(..., isect_capacity=K)`:

- default (two passes): a counting kernel, `torch.cumsum` over every Gaussian, a
  host read of the total, then an emitting kernel that re-reads every projected
  Gaussian and repeats the tile enumeration at its prefix-sum offset;
- single pass: one kernel counts a Gaussian's tiles, prefix-sums the counts
  across the warp, reserves the warp's contiguous range of a pre-allocated buffer
  of `K` pairs with one atomic on a global cursor, and writes the pairs there.
  The host still reads the final count (to size the sort), as before.
- Above `K` it warns (stderr) and falls back to the two passes. The caller owns
  `K` (PennyGS sizes it from the peak of earlier iterations).
- The pre-sort order depends on warp scheduling. After the (stable) radix sort,
  runs of exactly equal keys are put back in Gaussian index order by a small
  kernel, so sorted outputs equal the two-pass ones bit for bit. Equal keys are
  not rare: float32 depths collide (0.01-0.17 % of water2 pairs, 1.4 % in the
  garden test scene with co-located Gaussians).

## Results

`single_pass_benchmark.py --scope binning` (`isect_tiles` + `isect_offset_encode`,
capacity 1.25x the intersections, ms):

| View | Intersections | two passes | single pass | Speedup | Keys, ids, offsets |
|---|---:|---:|---:|---:|---|
| 0 | 1,011,765 | 0.307 | 0.257 | 1.195x | identical |
| 250 | 4,830,907 | 1.088 | 0.954 | 1.141x | identical |
| 500 | 996,736 | 0.306 | 0.242 | 1.264x | identical |
| 750 | 1,781,205 | 0.423 | 0.326 | 1.297x | identical |
| 1000 | 754,760 | 0.244 | 0.199 | 1.227x | identical |
| 1250 | 619,808 | 0.229 | 0.192 | 1.194x | identical |
| **geomean** | | | | **1.219x** | |

`--scope render` (full renderer forward + backward, ms): 16.27 -> 16.21,
26.57 -> 26.50, 16.50 -> 16.43, 17.93 -> 17.82, 16.10 -> 16.03, 15.79 -> 15.74
(geomean **1.004x**); images bit-identical, gradients within the run-to-run
noise of the atomics (max rel-L2 6.2e-05). Binning is 0.2-1.1 ms of a 16-27 ms
render here.

Measurements: `measurements/single_pass_{binning,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_single_pass_isect.py
PYTHONPATH=$PWD python benchmarks/pennygs/single_pass_benchmark.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --scope binning --output binning.json   # then --scope render
```
