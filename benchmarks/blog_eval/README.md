# CUDA optimization blog evaluation artifacts

The independent **post-PR9** water2 follow-up compares against `a7050c6`, not
PR9's older baseline. See [`water2_20261030/`](water2_20261030/README.md) for
retained/rejected experiments, raw timings and unchanged numerical gates.
`colmap_cameras.py` adapts PINHOLE COLMAP **text** models to this harness;
`run_followup.py` runs alternating serial jobs and records parity failures
without suppressing later timing jobs (it exits unsuccessfully if any remain).
The original summarizer/plotter consumes its `base-job*.json` / `pr-job*.json`.

This evaluation belongs to the gsplat repository; it does not require a parent
Gaussian-splatting checkout. The large scene PLY and `cameras.json` are external
inputs and are intentionally not committed. Supply them explicitly to the
harness (`--ply`, `--cameras`). The benchmark scripts use `plyfile`, PyTorch,
and the gsplat extension; use the environment already configured for this clone.

Example from the gsplat repository root:

```bash
python benchmarks/blog_eval/gsplat_blog_benchmark.py \
  --ply /path/to/point_cloud.ply --cameras /path/to/cameras.json \
  --extension /path/to/gsplat/csrc.so --max-points 0 \
  --compact-box --timing-only --warmup 30 --repeats 80
```

The data loader expects the supplied 3DGS PLY fields (`x/y/z`, `rot_*`,
`scale_*`, `opacity`, `f_dc_*`, optional `f_rest_*`) and a JSON camera list with
`rotation`, `position`, intrinsics and image dimensions. It converts the
camera-to-world extrinsics to world-to-camera, exponentiates log-scales and
sigmoids opacity logits before calling the renderer.

## Reference integrity

Use either `--save-reference baseline.pt` or `--compare-reference baseline.pt`,
never both in one invocation. Conflicting actions are rejected before extension
loading, scene loading or file writes, so a comparison cannot replace its own
baseline.

Single-pass reference protocol version **3** records SHA-256 hashes of both
input files and all selected rendering settings, including `tile_size`, `compact_box` and
`compact_box_impl`. Hashing is setup work outside the timed rendering loop.
The fixed-count gradient ensemble uses the distinct protocol
**`predeclared_raw_parameter_mean_v2`**, retaining its own script content hash.
Absolute input paths (and the ensemble script path) are local `provenance`, not
equality fields: identical bytes relocated to another directory remain compatible.
Both protocols check the CPU-loaded reference version first, then input hashes,
argument-known settings, and scene metadata (dimensions, point count and SH
degree) before loading the extension or uploading tensors. The same metadata
object is passed to the loader, avoiding a second PLY/camera parse; the resulting
tensor shapes are checked against it before rendering. The ensemble checks
upstream hashes and device identity before its first pass.

The preflight also validates the standard 3DGS vertex fields and SH schema:
required position/scale/rotation/opacity/DC fields, contiguous numeric
`f_rest_0...` names, three rest coefficients per basis, and a square basis count.
Malformed PLY layouts fail with a descriptive `ValueError` before CUDA setup,
rather than surfacing later as a property lookup or reshape error.

Regenerate the old single-pass v2 and ensemble v1 references with
`--save-reference` using the updated matching harness/settings, then compare with
`--compare-reference` in a separate process. There is no implicit conversion of
old references. Cross-setting ablations need their own matching baselines.

The raster fixture suite also appends channel-2 cases without changing the order
or index-derived seeds of the existing named fixtures; the saved suite config
still rejects references made with an earlier suite version.
The water2 follow-up appends eight tile32 staging-boundary cases, preserving
all 166 PR9 cases and their index-derived seeds. Regenerate both baseline and
candidate snapshots with the same updated suite (174 cases total).

The protocol and argument guards have CPU-only regression tests:

```bash
python -m pytest -q tests/test_blog_benchmark_reference.py tests/test_blog_ensemble_reference.py
```

The reduce-scatter helper also has a standalone ownership/sum test independent
of the rasterizer VJP. From the gsplat repository root, compile and run it with
the same CUDA toolkit used for the extension:

```bash
nvcc -std=c++17 -arch=sm_89 -Igsplat/cuda/include \
  tests/cuda/warp_reduce_scatter_contract.cu -o /tmp/scatter-contract
/tmp/scatter-contract
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/scatter-contract
```

It checks unique component owners and exact integer sums for dimensions 1–16,
with one and multiple warps per block. The raster matrix additionally exercises
the production 9–12 component layouts and the partial-warp fallback.

The canonical evaluation/tutorial is `docs/cuda_blog_zh/`. Final measurements
and correctly scoped limitations are in `REPORT.md` and
`results/REBUILT_FINAL_SUMMARY.md`; machine-readable final verification and
build identity are in `results/final_verification.json` and
`results/build_manifest.json`. Historical/retired-binary experiments remain
labelled as historical in the report; do not cite them as final results.

The harness supports full render and detached raster-only forward+backward,
with explicit `--extension` to load one CUDA binary per process. Baseline and
candidate extensions must be tested in separate processes because pybind CUDA
enum registrations collide when both are loaded into one interpreter. See
`docs/cuda_blog_zh/gsplat_cuda_tutorial.pdf` and the adjacent chapter sources
for the protocol, derivations and examples.
