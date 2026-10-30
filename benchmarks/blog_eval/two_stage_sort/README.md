# Two-stage visibility/depth/tile sorting: independent experiment

**Status: experimental, opt-in, not a demonstrated end-to-end speedup.**
This work branches directly from default `dev/dc` at `a7050c6` (merged PR9).
It is independent of PR11 and does not include PR11's color-staging changes.

## Code changes

The CUDA blog's two-stage approach replaces sorting all `I` full intersection
keys by a smaller depth sort followed by a narrow stable tile sort:

1. Reuse the existing per-Gaussian compact-box tile counts.
2. Stably select `M` Gaussian indices with positive tile counts, preserving
   original flatten order (including interleaved packed camera inputs).
3. Gather unsigned raw positive-float depth bits and stably sort 32-bit keys.
4. Gather tile counts in that order and prefix-sum to assign emission offsets.
5. Reuse the existing tile/compact-box emission kernel, emitting 16-bit encoded
   camera/tile keys with full 32-bit Gaussian IDs in depth-sorted order.
6. Stable-sort those narrow tile keys. Reconstruct the public 64-bit
   `(camera | tile | depth)` keys for the existing offset/rendering API.

Exact depth ties retain original flatten order. No depth quantization, 24-bit
Gaussian-ID packing, compact-box changes, or raster kernel changes are used.
The original sorting entry point remains intact. Select the experiment with
`rasterization(..., two_stage_sort=True)` or
`isect_tiles(..., two_stage_sort=True)`; the default is **False**.

Eligibility checks the maximum **encoded camera/tile ID**, not merely
`camera_count * tile_count`: the existing power-of-two camera stride can exceed
16 bits even if that product fits. Oversized namespaces, unsupported types or
sizes, invalid live depths/camera IDs, and `sort=False` retain the original
path. Negative/zero/nonfinite-depth fallback preserves baseline intersection
bytes; it does not make malformed baseline inputs valid for downstream rendering.

Changed implementation files:

- `gsplat/cuda/csrc/isect_tiles.cu`: compaction, two stable sorts, shared emission,
  key reconstruction, and guarded fallback.
- `gsplat/cuda/include/bindings.h`, `gsplat/cuda/csrc/ext.cpp`: separate
  `isect_tiles_two_stage` entry, retaining the original binding/signature.
- `gsplat/cuda/_wrapper.py`, `gsplat/rendering.py`: opt-in selection.
- `tests/test_two_stage_sort.py`: exact output/order/fallback contract tests.
- `benchmarks/blog_eval/benchmark_two_stage_sort.py`: binning API timing,
  radix-kernel profiling, exact intersection references, and PR9 full-render
  image/gradient checks. It reuses PR9 helpers without changing their gates.

## Verified smoke measurement

Local RTX 4090/SM89, PyTorch 2.10.0+cu128, CUDA toolkit12.8. Supplied water2
iteration020000 scene: all **1,932,132** degree-1 SH Gaussians. View1247 from
the supplied COLMAP-derived camera list, native **1280×720**, tile16,
compact-box sweep. It has **1,049,168 visible Gaussians** and **2,909,623
intersections**: `M/N≈54%`, not the article's `M ≪ N` regime.

| Metric | PR9 baseline | Two-stage candidate | Observation |
|---|---:|---:|---|
| Radix kernels, summed GPU time per call | 0.266 ms | 0.181 ms | About32% less sorting GPU time |
| Complete binning API, CUDA-event mean | 0.541 ms | 0.634 ms | About17% slower |
| Complete binning API, synchronized wall mean | 0.617 ms | 0.651 ms | About5.5% slower |
| Full renderer inference mean | 3.697 ms | 3.839 ms | About3.8% slower |
| Full renderer forward+backward mean | 9.074 ms | 9.217 ms | About1.6% slower |

This is **one preliminary job per variant**, 10 warmups/20 timed samples, not a
three-job/all-view study. Radix figures come from a separate five-call profiler
diagnostic and exclude compaction, scans, gathering, emission, reconstruction,
and synchronization. API CUDA-event times include launch/synchronization gaps;
wall times include the synchronized host call. These are different timing
scopes: **do not subtract profiler radix time from API latency to infer exact
component costs**. Raw samples, profiler kernel names and parity are retained
in [`smoke-base.json`](smoke-base.json) and
[`smoke-candidate.json`](smoke-candidate.json).

The article's reported79% sorting reduction is not reproduced here. Moreover,
the existing gsplat baseline already limits the sorted key range to
`32 + tile_bits + camera_bits`—**45 bits in this case**, not necessarily eight
passes across all64 bits. The precise pass policy is CUB/hardware dependent.

## Why faster radix kernels did not improve total latency

The new pipeline adds a validation/compaction pass, index/depth/count gathers,
sorted-count scan, and public64-bit-key reconstruction. This implementation
also reads compacted size/validity back to the CPU, then reads the intersection
total after the sorted scan, introducing two host synchronization points versus
the baseline's one intersection-total readback. `M` is still large in this
scene. Those costs can erase the radix-only savings.

These are mechanism candidates supported by the implemented operation graph,
**not measured per-stage bottleneck attribution**. Follow-up work would need
separate stage timing and broader workloads before changing defaults.

## Correctness and scope

- The measured scene's per-Gaussian counts, sorted64-bit intersection keys,
  and32-bit flatten IDs are **exactly equal to the original binary**.
- Full-render image/alpha are bit-identical. All six full-parameter/absgrad
  comparisons pass the unchanged PR9 `1e-4` relative-L2 gate; maximum is
  **1.39e-6** (raw quaternion). Image/alpha gate remains `2e-5`.
- **47 CUDA contract tests pass**, covering stable ties/neighboring float bits,
  dense/packed inputs, compact-box modes, empty/single inputs, encoding
  boundaries, invalid-depth and float64 fallback, unsorted mode, both CUB buffer
  modes, a non-default stream, and Gaussian IDs beyond24 bits.
- **96 existing CPU protocol/default tests pass** (143 selected tests total).
- No sanitizer run, broader repeated scene sweep, 4K performance result,
  training-convergence claim, or full-repository test claim is made. Work was
  deliberately stopped at a recorded experimental PR rather than claiming a
  speedup from this smoke run. The original path remains the default.

## Reproduce

Use an existing compatible environment and build both the default-branch
baseline and this candidate with `CUDA_HOME=/usr/local/cuda-12.8`,
`TORCH_CUDA_ARCH_LIST=8.9`. Example candidate build:

```bash
python setup.py build_ext --build-lib build/two_stage --build-temp build/objects
cp build/two_stage/gsplat/csrc.so gsplat/csrc.so  # local worktree only; no install
PYTHONPATH="$PWD" python -m pytest -q tests/test_two_stage_sort.py

# One fresh process per extension. Supply COLMAP-derived camera JSON, not the
# supplied scene's mismatched adjacent cameras.json. The original binary can
# be loaded by this PR's wrapper when --two-stage is absent.
PYTHONPATH="$PWD" python benchmarks/blog_eval/benchmark_two_stage_sort.py \
  --extension /path/to/a7050c6/csrc.so --ply /path/to/point_cloud.ply \
  --cameras /path/to/colmap-derived-cameras.json --camera-indices 1247 \
  --resolutions 1280x720 --warmup 10 --repeats 20 \
  --save-reference-dir /tmp/opencode/sort-new-refs --output /tmp/opencode/base.json
PYTHONPATH="$PWD" python benchmarks/blog_eval/benchmark_two_stage_sort.py \
  --extension build/two_stage/gsplat/csrc.so --two-stage \
  --ply /path/to/point_cloud.ply --cameras /path/to/colmap-derived-cameras.json \
  --camera-indices 1247 --resolutions 1280x720 --warmup 10 --repeats 20 \
  --compare-reference-dir /tmp/opencode/sort-new-refs --output /tmp/opencode/candidate.json
```

Tensor references and compiled binaries are external artifacts, not committed.
Binary/input identities are in the raw reports; the compact verification file
records the source identities and final selected-test result.
