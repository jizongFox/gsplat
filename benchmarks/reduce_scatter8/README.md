# S02 <=8: actual DC + Gaussian normals renderer

Standalone harness; imports neither the parent application nor PR9/PR13 code.
The CPU tests read the parent's formula ASTs without importing its dependencies
or `torch.compile`. No source file is changed in either selected gsplat checkout.
Prebuilt extensions are required; the benchmark does not build/JIT anything.

## Exact path

World-space Gaussian means; normalized **wxyz** quaternion; exponentiated PLY
scales; sigmoid PLY opacity. RGB is `SH2RGB(f_dc)` without view-dependent SH.
The normal is the quaternion rotation **column selected by scale.argmin**, not
local +Z. It is transformed by world-to-camera rotation, divided by its norm,
then has **+1e-5 added after the division**, sanitized, and multiplied by
`sign(dot(normal, R @ mean + t))`. The source convention has nonnegative dot
product after orientation; sign(0) produces a zero normal. This preserves the
source's convention rather than imposing a different interpretation of
"face camera".

The raster call receives **RGB + normal xyz = 6 colors**. `RGB+ED` appends camera
depth, making **7 logical channels**, padded to **8 physical channels** by the
wrapper. The S02 component count is `8 + 3 + 2 + 1 + 2 = 16`.
The defaults match `gsplat_renderer_3d_normals.render`: unpacked, pinhole,
`absgrad=True`, compact-box sweep, zero background, RGB clamp, camera-coordinate
rendered normal normalization and depth-derived surface normal calculation.
The renderer's Python output container and gradient-source closure are omitted.
Compiled parent normal decorators are deliberately omitted to keep this harness
dependency-free; CUDA event timing includes the eager formula launches.

Upstream is CPU-seeded random **[RGB + expected depth: 4 channels] + alpha**,
transferred unchanged to CUDA. There is no artificial normal-map loss. All model
leaf gradients (world means, raw quaternion, log scales, opacity logits, DC),
signed projected mean gradients and projected absolute gradients are compared.
Outputs include both normal maps, even though their gradients are not seeded.
Fixed PR9 limits: forward max-abs `2e-5`, every gradient relative-L2 `1e-4`.
Baseline/candidate local repeats must pass the same limits; NaN/Inf fails.

References bind PLY/camera content, harness content, all rendering settings,
case identities and CPU upstream hashes; stored tensor hashes are validated.
Save/compare are mutually exclusive. Existing references cannot be overwritten.
Reference and JSON writes inside `--repo` are forbidden. All cameras are measured
by default, separately in world coordinates. `--compact-box both` runs sweep-on
and compact-off cases. Results contain forward/backward/combined CUDA-event
samples, medians, means, tensor hashes, comparison errors and extension hash.
Candidate cases also report baseline/candidate median speedups for all three
timing stages (greater than one is faster). CLI stdout reports those medians.

Native renderer parity is especially strong: image+ED, alpha, rendered normals,
and depth-derived surface normals are bit-identical for views 0, 623 and 1247;
maximum gradient relative-L2 is 9.54e-7, 3.97e-6 and 1.86e-6 respectively,
all below the unchanged PR9 1e-4 gate. See `REPORT.md`,
`native_accuracy_speed.csv`, and `plots/native_accuracy_speed.png`.

## Commands

Run from this worktree. Choose a Python environment containing torch, plyfile,
pytest and gsplat's Python dependencies. CUDA must match that environment's
prebuilt extensions. Neither benchmark invocation below has been GPU-tested.

```bash
PY=/home/jizong/workspace/dConstruct/gaussian-splatting/venv/bin/python
HARNESS=/tmp/opencode/gsplat-reduce-scatter8/benchmarks/reduce_scatter8
EVIDENCE=/tmp/opencode/reduce-scatter8-renderer-evidence
mkdir -p "$EVIDENCE"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -p no:cacheprovider "$HARNESS/test_renderer_cpu.py"

# Compile only: existing helper instantiates DIM=1..16, including full DIM=16.
# Do not execute the resulting CUDA binary in a CPU-only validation session.
/usr/local/cuda-12.8/bin/nvcc -std=c++17 -arch=sm_86 -Igsplat/cuda/include tests/cuda/warp_reduce_scatter_contract.cu -o "$EVIDENCE/scatter-contract"

# For COLMAP binary models, first export a TEXT model using COLMAP model_converter.
# Only undistorted PINHOLE/SIMPLE_PINHOLE cameras are accepted, never silently approximated.
PYTHONDONTWRITEBYTECODE=1 "$PY" "$HARNESS/convert_colmap.py" --cameras-txt /path/to/cameras.txt --images-txt /path/to/images.txt --output "$EVIDENCE/cameras.json"

# Run these as SEPARATE processes, using the SAME harness and scene/settings.
# Replace BASELINE_REPO, BASELINE_SO, CANDIDATE_SO, and PLY with actual paths.
CUDA_HOME=/usr/local/cuda-12.8 TORCH_EXTENSIONS_DIR="$EVIDENCE/baseline-cache" PYTHONDONTWRITEBYTECODE=1 "$PY" "$HARNESS/renderer_benchmark.py" --repo "$BASELINE_REPO" --extension "$BASELINE_SO" --ply "$PLY" --cameras "$EVIDENCE/cameras.json" --compact-box both --save-reference "$EVIDENCE/baseline.pt" --output "$EVIDENCE/baseline.json"

CUDA_HOME=/usr/local/cuda-12.8 TORCH_EXTENSIONS_DIR="$EVIDENCE/candidate-cache" PYTHONDONTWRITEBYTECODE=1 "$PY" "$HARNESS/renderer_benchmark.py" --repo /tmp/opencode/gsplat-reduce-scatter8 --extension "$CANDIDATE_SO" --ply "$PLY" --cameras "$EVIDENCE/cameras.json" --compact-box both --compare-reference "$EVIDENCE/baseline.pt" --output "$EVIDENCE/candidate.json"
```
