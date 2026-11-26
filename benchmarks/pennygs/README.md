# PennyGS (arXiv 2611.27525) contributions on gsplat

Each PennyGS contribution is a separate opt-in change with its own PR and its own
numbers, measured against this branch's base (`dev/dc` at `29ee2a0`) in one
process on one build, so the variants share projection, intersections, forward
pass and upstream gradients.

## Workload

- RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.9`.
- water2 3DGS model with 2,909,851 Gaussians (DC colour), 1280x720 pinhole views
  from its `cameras.json` (views 0, 250, 500, 750, 1000, 1250). The PLY and cameras
  are external inputs, not committed.
- Renderer shape of the parent application (`gsplat_renderer_3d_normals`):
  RGB + camera-space normal + expected depth = 7 logical channels, padded to 8;
  unpacked, absgrad, compact-box sweep, tile 16
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`).
- `backward_variants.py --scope raster`: CUDA events around the backward of
  `rasterize_to_pixels` alone, with fixed random upstream gradients.
  `--scope render`: the full renderer forward + backward.
- 20 warmups and 80 timed repeats; variants alternate inside each repeat; medians.
- Gate (unchanged PR9 limit): every gradient within relative L2 1e-4 of the
  default `"pixel"` kernel, and identical forward outputs.

## Lean backward state (`backward="pixel_lean"`, PennyGS Sec. 3.6)

In the backward recurrence the colours behind a Gaussian enter `dL/dalpha` only
through their dot product with the pixel's colour gradient:

```text
dL/dalpha_i = T_i * (c_i . v) - (sum_{j behind i} c_j * alpha_j * T_j) . v / (1 - alpha_i) + ...
```

so a pixel can carry the scalar `sum_j (c_j . v) alpha_j T_j` instead of one
partial sum per channel. Same math; only the summation order of that term
changes. For the 8-channel kernel this drops registers from 72 to 64 per
thread, which lets four 256-thread blocks instead of three reside on an SM.

| View | Intersections | `pixel` backward (ms) | `pixel_lean` (ms) | Speedup | Max grad rel-L2 |
|---|---:|---:|---:|---:|---:|
| 0 | 1,011,765 | 1.258 | 1.175 | 1.070x | 9.3e-07 |
| 250 | 4,830,907 | 6.923 | 6.974 | 0.993x | 2.0e-06 |
| 500 | 996,736 | 1.362 | 1.239 | 1.100x | 7.0e-07 |
| 750 | 1,781,205 | 2.243 | 2.184 | 1.027x | 8.8e-07 |
| 1000 | 754,760 | 1.270 | 1.186 | 1.071x | 1.4e-06 |
| 1250 | 619,808 | 1.027 | 0.952 | 1.079x | 5.1e-07 |
| **geomean** | | | | **1.056x** | |

Full renderer forward + backward: 16.28 -> 16.20, 26.57 -> 26.64, 16.50 -> 16.35,
18.02 -> 17.96, 16.17 -> 16.08, 15.68 -> 15.62 ms (geomean **1.004x**); all
gradients pass (max rel-L2 6.6e-05). The raster backward is 1-7 ms of a 16-27 ms
render here; projection, normals and the other per-Gaussian work dominate.

Measurements: `measurements/lean_{raster,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_backward_variants.py
PYTHONPATH=$PWD python benchmarks/pennygs/backward_variants.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --views 0 250 500 750 1000 1250 --scope raster \
  --warmup 20 --repeats 80 --output raster.json      # then --scope render
```
