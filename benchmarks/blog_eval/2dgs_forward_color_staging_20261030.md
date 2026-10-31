# 2DGS forward color-staging candidate: measured limitations

This PR deliberately preserves the evaluated candidate even though it is **not a demonstrated improvement**. It is an experiment, not a production recommendation.

## Context

PR9 (`a7050c6`) changes only the 3DGS backward rasterizer; PR11 (`eed7e91`, still open at the time of evaluation) changes only the 3DGS forward rasterizer. The 2DGS renderer has separate kernels. This PR applies the analogous forward staging idea to the 2DGS forward kernel only: for `COLOR_DIM <= 4`, stage the Gaussian's colors in shared memory with its geometry, then load the staged color for each pixel hit. Wider channel specializations retain the existing global-load path. Blend math and thresholds are unchanged.

## Evaluation

Workload: full supplied COLMAP scene (20,789,814 points), cameras `[0, 2552, 5103]`, 1280×720, FoV-preserving intrinsics, tile size 16, unpacked rasterization, Compact Box sweep, absgrad, `RGB+ED`. RTX 4090, existing Python environment (PyTorch 2.10.0+cu128 / CUDA 12.8). Three serial jobs per binary; each timing used 2 warmups and 3 repeats. These small repeat counts make this an exploratory result, not a confidence interval. The recorded measurements were taken against PR11 head `eed7e91`; the 2DGS source file was unchanged between that head and this PR's base, and this PR applies the same staged-color source delta. On this rebased PR head, the modified 2DGS source SHA-256 is `099be7845c4796479ee8dce341cc57071c1a6ef7154124f7052137d4bec9cf19`.

| Variant | Inference jobs (ms) | Forward jobs (ms) |
|---|---|---|
| Baseline | 205.593, 200.334, 202.984 | 207.878, 199.805, 202.658 |
| Candidate | 205.797, 214.779, 205.625 | 216.110, 212.520, 209.576 |

The candidate is slower in forward in all three jobs (median approximately 6.3% regression); inference is roughly flat to slower (median approximately 2.1% regression). No backward speedup is claimed.

## Strict forward parity failure

Baseline repeated output was bit-identical. Candidate-vs-baseline on the same full scene/cameras had these maximum absolute differences:

| Output | Max absolute difference | Existing gate |
|---|---:|---:|
| Render colors (RGB+ED) | 0.0040913 | 2e-5 |
| Alpha | 7.8797e-5 | 2e-5 |
| Render normals | 0.0067205 | 2e-5 |
| Distortion | 0.0867317 | 2e-5 |
| Median depth | 0.0021689 | 2e-5 |
| Median IDs | 0.0459244 | exact/discrete |

These are actual failures; no tolerance was relaxed. As a result, this candidate does not pass the established forward parity gate.

## OOM limits and reproducibility

The full-scene, 3-camera 1280×720 backward attempt OOM'd (the failing allocation requested 714 MiB with roughly 22 GiB already resident). Full-scene 4K inference also OOM'd (a 5.95 GiB allocation). We did not substitute a smaller scene and do not claim these checks passed. The local benchmark harness and raw timing/output files were not committed; they were run from `/tmp/opencode/bench_2dgs_scene.py` and `/tmp/opencode/2dgs-*.json` in the evaluation environment. The scene remains at the user's local COLMAP path and is not bundled here.

## Decision

The change is submitted exactly as evaluated for review/experimentation, despite being slower and failing parity. It is not production-safe, parity-certified, or recommended for use until the numerical mismatch is understood and fixed and the performance is re-evaluated. The focused `tests/test_2dgs.py` suite passes 17 tests when suppressing its pre-existing `PytestRemovedIn9Warning` during collection; without that warning filter, pytest 9 aborts collection at an existing fixture-mark warning.
