# PennyGS (arXiv 2611.27525) Sec. 3.3: 32-bit keys + single-pass emission, combined

Speed experiment only. It merges jizongFox/gsplat#23 (32-bit sort keys) and #24
(single-pass key emission), both from `dev/dc` at `29ee2a0`, to measure them
together. #23 and #24 stay the reviewable units.

Workload: the same as #23 and #24. RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8;
water2 3DGS model, 2,909,851 DC-colour Gaussians; 1280x720 views 0, 250, 500,
750, 1000, 1250 of its `cameras.json` (external inputs, not committed); the
parent application's renderer shape (RGB + camera normal + expected depth, 7
channels padded to 8, absgrad, compact-box sweep, tile 16). One process, the
five variants alternate inside each repeat, 20 warmups, 80 repeats, medians.
Capacity for the single pass: 1.25x the intersections.

## Variants

| Arm | Keys | Emission | Equal-key fix-up | Output |
|---|---|---|---|---|
| A `default` | 64-bit | two passes | - | reference |
| B `short_keys` (#23) | 32-bit | two passes | - | approximate depth order |
| C `single_pass` (#24) | 64-bit | one pass | yes | identical to A |
| D `combined` | 32-bit | one pass | yes | identical to B |
| E `combined_no_ties` | 32-bit | one pass | **no** | same pairs as B; order inside a run of equal keys depends on scheduling |

`isect_tiles(..., short_keys=True, single_pass_capacity=K,
single_pass_restore_ties=...)`, or `rasterization(..., isect_short_keys=True,
isect_capacity=K, isect_restore_ties=...)`. `single_pass_restore_ties` is new
and only exists for this experiment.

## Binning

`isect_combined_benchmark.py --scope binning` (`isect_tiles` +
`isect_offset_encode`, ms):

| View | Intersections | A default | B short keys | C single pass | D combined | E combined, no fix-up |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,011,765 | 0.312 | 0.259 (1.21x) | 0.257 (1.21x) | 0.212 (1.47x) | 0.202 (1.55x) |
| 250 | 4,830,907 | 1.073 | 0.597 (1.80x) | 0.937 (1.14x) | 0.550 (1.95x) | 0.444 (2.41x) |
| 500 | 996,736 | 0.311 | 0.259 (1.20x) | 0.243 (1.28x) | 0.207 (1.50x) | 0.196 (1.59x) |
| 750 | 1,781,205 | 0.424 | 0.343 (1.24x) | 0.327 (1.30x) | 0.254 (1.67x) | 0.234 (1.81x) |
| 1000 | 754,760 | 0.247 | 0.211 (1.17x) | 0.200 (1.23x) | 0.171 (1.44x) | 0.158 (1.56x) |
| 1250 | 619,808 | 0.230 | 0.204 (1.13x) | 0.193 (1.19x) | 0.161 (1.43x) | 0.154 (1.49x) |
| **geomean** | | 1.00x | **1.27x** | **1.23x** | **1.57x** | **1.71x** |

- B and C reproduce #23 (1.266x) and #24 (1.219x).
- The two gains multiply: 1.27 x 1.23 = 1.56, measured 1.57x.
- Every arm gives the same tile ranges as A. C equals A and D equals B in keys,
  Gaussian ids and offsets; E has B's keys and pairs, not its order inside runs.

### The equal-key fix-up costs more with 32-bit keys

After the single pass, runs of exactly equal keys are put back in Gaussian
index order (one thread per run, insertion sort). With 32-bit keys only 19 depth
bits remain here, so many more pairs share a key:

| View | 64-bit: pairs in runs | max run | 32-bit: pairs in runs | mean run | p99 run | max run | fix-up (D - E, ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02 % | 2 | 52.4 % | 2.6 | 7 | 27 | 0.010 |
| 250 | 0.34 % | 4 | 88.8 % | 6.0 | 35 | 213 | 0.105 |
| 500 | 0.07 % | 3 | 63.8 % | 3.2 | 13 | 38 | 0.011 |
| 750 | 0.08 % | 2 | 72.7 % | 3.4 | 13 | 56 | 0.020 |
| 1000 | 0.14 % | 2 | 56.1 % | 2.7 | 8 | 44 | 0.013 |
| 1250 | 0.07 % | 2 | 58.7 % | 2.8 | 8 | 31 | 0.007 |

The fix-up is 0.01-0.02 ms on most views, but 0.105 ms on view 250 (the densest,
4.8M pairs, runs up to 213), where it takes back about half of what the single
pass saves.

## Full renderer

`--scope render` (forward + backward, ms), two separate runs:

| View | A default | B short keys | C single pass | D combined | E combined, no fix-up |
|---|---:|---:|---:|---:|---:|
| 0 | 16.30 | 16.25 (1.003x) | 16.21 (1.006x) | 16.16 (1.009x) | 16.22 (1.005x) |
| 250 | 26.55 | 26.15 (1.015x) | 26.39 (1.006x) | 26.11 (1.017x) | 26.03 (1.020x) |
| 500 | 16.49 | 16.46 (1.002x) | 16.43 (1.004x) | 16.39 (1.006x) | 16.39 (1.006x) |
| 750 | 17.99 | 17.90 (1.005x) | 17.89 (1.005x) | 17.82 (1.009x) | 17.79 (1.011x) |
| 1000 | 16.21 | 16.15 (1.003x) | 16.09 (1.007x) | 16.10 (1.006x) | 16.07 (1.008x) |
| 1250 | 15.80 | 15.73 (1.005x) | 15.72 (1.005x) | 15.69 (1.007x) | 15.68 (1.008x) |
| **geomean, run 1** | 1.000x | **1.0057x** | **1.0055x** | **1.0092x** | **1.0098x** |
| **geomean, run 2** | 1.000x | **1.0072x** | **1.0071x** | **1.0088x** | **1.0113x** |
| forward only, run 1 | 1.000x | 1.017x | 1.016x | 1.028x | 1.030x |

- Combined (D) is **0.9 %** faster end to end in both runs; all of it is in the
  forward, where binning lives. The backward is unchanged.
- Without the fix-up (E) it is 1.0-1.1 %, but the output is no longer
  deterministic.
- Images and gradients (vs A): D is identical to B (RGB 47-59 dB from A,
  per-Gaussian gradients 10-16 % relative L2, the #23 approximation). C is
  bit-identical to A in images (gradients within atomics noise, at most
  2.1e-05). E differs from B by up to 0.13 in a few pixels.

## Conclusion

The two contributions combine: binning 1.57x, but the full renderer only
0.9 % because binning is 0.2-1.1 ms of a 16-27 ms render on this workload.
This is just under the parent application's 1 % bar, and the combined path
inherits #23's approximation (3-16 % gradient change). Not proposed for
merging.

## Correctness

- `tests/test_isect_combined.py`, 4 passed: combined equals the two-pass short
  keys (unpacked and packed), same keys and pairs without the fix-up, fallback
  above capacity, identical render.
- `tests/test_short_keys.py` and `tests/test_single_pass_isect.py`: 8 passed.
- Compute Sanitizer memcheck on `tests/test_isect_combined.py`: 0 errors.
- `tests/test_basic.py` and `tests/test_rasterization.py`: the same 18
  failures as on `29ee2a0`.

Measurements: `measurements/isect_combined_{binning,render,render_r2}.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_isect_combined.py
PYTHONPATH=$PWD python benchmarks/pennygs/isect_combined_benchmark.py --ply <point_cloud.ply> \
    --cameras <cameras.json> --scope binning --output binning.json
PYTHONPATH=$PWD python benchmarks/pennygs/isect_combined_benchmark.py --ply <point_cloud.ply> \
    --cameras <cameras.json> --scope render --output render.json
```
