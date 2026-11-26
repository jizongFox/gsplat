# PennyGS (arXiv 2611.27525) contributions on gsplat

Each PennyGS contribution is a separate opt-in change with its own PR and its own
numbers against `dev/dc` at `29ee2a0`, measured in one process on one build with
the variants alternating inside each repeat (20 warmups, 80 repeats, medians).

Workload: RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.9`;
water2 3DGS model, 2,909,851 DC-colour Gaussians; 1280x720 pinhole views 0, 250,
500, 750, 1000, 1250 of its `cameras.json` (external inputs, not committed); the
parent application's renderer shape (RGB + camera normal + expected depth, 7
channels padded to 8, absgrad, compact-box sweep, tile 16;
`benchmarks/reduce_scatter8/renderer_benchmark.py::render`).

## 32-bit sort keys (`isect_tiles(short_keys=True)`, `rasterization(isect_short_keys=True)`, Sec. 3.3)

The default intersection id is 64 bits: camera | tile | 32 depth bits, sorted over
`32 + Xt + Xc` bits (45 here). Short keys pack camera | tile | the top
`32 - Xt - Xc` bits of the depth's float bits (sign dropped) into 32 bits: 19
depth bits here, a relative depth step of 2^-11. The sort runs over 32-bit keys
(4 radix passes instead of 6, half the key bytes); `isect_offset_encode` decodes
either key type by dtype. Gaussians whose depths agree in the kept bits keep
their input order instead of depth order, so this is an approximation.
At least 16 depth bits are required (otherwise `RuntimeError`).

`short_keys_benchmark.py --scope binning` (`isect_tiles` + `isect_offset_encode`, ms):

| View | Intersections | default | short keys | Speedup | Tile ranges |
|---|---:|---:|---:|---:|---|
| 0 | 1,011,765 | 0.307 | 0.259 | 1.186x | identical |
| 250 | 4,830,907 | 1.084 | 0.603 | 1.798x | identical |
| 500 | 996,736 | 0.318 | 0.272 | 1.167x | identical |
| 750 | 1,781,205 | 0.417 | 0.339 | 1.230x | identical |
| 1000 | 754,760 | 0.244 | 0.209 | 1.167x | identical |
| 1250 | 619,808 | 0.235 | 0.204 | 1.156x | identical |
| **geomean** | | | | **1.266x** | |

`--scope render` (full renderer forward + backward, ms) and the approximation:

| View | default | short keys | Speedup | RGB PSNR vs default | Max abs | Gradient rel-L2 (max over leaves) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 16.26 | 16.18 | 1.005x | 52.3 dB | 0.192 | 0.10 |
| 250 | 26.49 | 26.07 | 1.016x | 53.5 dB | 0.189 | 0.11 |
| 500 | 16.42 | 16.37 | 1.003x | 47.4 dB | 0.269 | 0.13 |
| 750 | 17.95 | 17.89 | 1.004x | 50.2 dB | 0.232 | 0.10 |
| 1000 | 16.13 | 16.08 | 1.003x | 59.1 dB | 0.162 | 0.10 |
| 1250 | 15.81 | 15.77 | 1.002x | 51.7 dB | 0.169 | 0.16 |
| **geomean** | | | **1.006x** | | | |

Reordered near-equal-depth Gaussians change a few pixels visibly (max abs ~0.2)
and the per-Gaussian gradients by 3-16 % relative L2 (DC colour most, opacity
least); whether training absorbs that was not measured. The binning stage is
0.2-1.1 ms of a 16-27 ms render on this workload.

Measurements: `measurements/short_keys_{binning,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_short_keys.py
PYTHONPATH=$PWD python benchmarks/pennygs/short_keys_benchmark.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --scope binning --output binning.json   # then --scope render
```
