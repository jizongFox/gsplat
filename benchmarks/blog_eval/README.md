# CUDA optimization blog evaluation artifacts

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
