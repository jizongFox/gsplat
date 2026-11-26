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

## Per-chunk backward traversal (`backward="chunk*"`, PennyGS Sec. 3.6) — experiment, slower here

One 128-thread block (4 warps) per 16x16 tile. A warp owns one Gaussian at a time
and walks the tile's eight 8x4-pixel chunks, its 32 lanes on the chunk's 32
pixels; pixel state (transmittance, behind-colour sums, colour gradient) lives in
shared memory. Back-to-front order per pixel is kept by per-chunk counters: a
Gaussian enters a chunk when every earlier Gaussian walking that chunk has left
(acquire/release on the counter, the warp barrier orders the lanes' writes). A
Gaussian's gradient is reduced once per tile (one set of atomics per Gaussian
and tile instead of per warp). Variants: `chunk` (standard arithmetic),
`chunk_lean` (scalar behind-colour state), `chunk_cull` (a Gaussian walks and
waits only for chunks its alpha >= 1/255 ellipse can reach: a conservative
bounding-box test whose threshold includes a bound on the float32 rounding of
the renderer's sigma over the tile; ill-conditioned, degenerate or nonfinite
conics walk every chunk) and
`chunk_lean_cull` (both, the paper's full design). Tile size 16 and at most 8
padded channels.

Raster backward (ms, speedup vs `pixel`; all gradients within rel-L2 2e-6):

| View | Intersections | `pixel` | `pixel_lean` | `chunk` | `chunk_lean` | `chunk_cull` | `chunk_lean_cull` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,011,765 | 1.255 | 1.170 (1.07x) | 2.145 (0.59x) | 1.635 (0.77x) | 1.833 (0.68x) | 1.351 (0.93x) |
| 250 | 4,830,907 | 6.881 | 6.958 (0.99x) | 25.216 (0.27x) | 24.864 (0.28x) | 14.517 (0.47x) | 14.528 (0.47x) |
| 500 | 996,736 | 1.363 | 1.240 (1.10x) | 2.414 (0.56x) | 1.717 (0.79x) | 2.010 (0.68x) | 1.423 (0.96x) |
| 750 | 1,781,205 | 2.245 | 2.190 (1.03x) | 4.390 (0.51x) | 3.486 (0.64x) | 3.311 (0.68x) | 2.611 (0.86x) |
| 1000 | 754,760 | 1.268 | 1.184 (1.07x) | 2.247 (0.56x) | 1.693 (0.75x) | 1.951 (0.65x) | 1.444 (0.88x) |
| 1250 | 619,808 | 1.027 | 0.950 (1.08x) | 1.756 (0.58x) | 1.208 (0.85x) | 1.537 (0.67x) | 1.057 (0.97x) |
| **geomean** | | 1.000x | 1.056x | 0.498x | 0.641x | 0.634x | 0.822x |

Full renderer forward + backward with `chunk_lean_cull`: geomean **0.952x**
(view 250: 26.56 -> 34.30 ms).

Why it loses here (resource usage from `cuobjdump --dump-resource-usage`, 8 channels):

| Kernel | Registers | Shared memory / block | Resident warps / SM |
|---|---:|---:|---:|
| `pixel` (256 threads) | 72 | 15 KB dynamic | 24 |
| `pixel_lean` | 64 | 15 KB dynamic | 32 |
| `chunk` (128 threads) | 80 | 27 KB | 12 |
| `chunk_lean_cull` | 78 | 21 KB | 16 |

- Eight channels make the per-pixel colour gradient and behind-colour state
  large (PennyGS has three colour channels and ~9 KB per block), so shared memory
  caps residency at 12-16 warps.
- Only four Gaussians of a tile are in flight at a time; on the densest view
  (4.8M intersections, ~1,300 per tile) that serial chain dominates (0.27-0.47x).
- The baseline is not upstream gsplat: `dev/dc` already has warp reduce-scatter
  and tile-tail trimming (PR9/PR14), which removed most of the atomic pressure
  this design targets. PennyGS's 3.49x is against upstream gsplat.
- One tuning attempt (64 Gaussians per batch, colour gradient read through L1
  instead of shared memory: 8.6 KB, 72 registers, 28 warps) did not change the
  outcome: `chunk_lean_cull` 0.90x / 0.78x / 1.04x on views 0 / 600 / 1200
  (`measurements/chunk_tuning_attempt_raster.json`); it is not in this branch.

Sanitizers: memcheck and synccheck 0 errors over `tests/test_backward_variants.py`
(34 cases). racecheck reports potential hazards on the shared pixel state
`s_T`/`s_buf`: that is the intended cross-warp handoff, ordered by the
acquire/release counter, which racecheck does not model (it tracks barriers
only). Every variant still matches the per-pixel gradients under the
sanitizer-perturbed scheduling.

Measurements: `measurements/chunk_{raster,render}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_backward_variants.py
PYTHONPATH=$PWD python benchmarks/pennygs/backward_variants.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --views 0 250 500 750 1000 1250 --scope raster \
  --warmup 20 --repeats 80 --output raster.json      # then --scope render
```
