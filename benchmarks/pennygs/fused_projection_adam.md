# PennyGS (arXiv 2611.27525) Sec. 3.7: fused projection backward + sparse Adam

Part of the PennyGS series on gsplat: each contribution is a separate opt-in change
measured against `dev/dc` at `29ee2a0` in one process on one build, variants
alternating inside each repeat (20 warmups, 80 repeats, medians).

Workload: RTX 4090 (SM 8.9), PyTorch 2.10 + CUDA 12.8; water2 3DGS model,
2,909,851 Gaussians; 1280x720 views 0, 250, 500, 750, 1000, 1250 of its
`cameras.json` (external inputs, not committed); random upstream gradients of the
projected means, depths and conics.

## Change

`gsplat.cuda._wrapper.fused_projection_adam(means, quats, log_scales, viewmat, K,
width, height, radii, conics, v_means2d, v_depths, v_conics, moments=..., lrs=...)`,
one pinhole camera, one thread per Gaussian:

- a Gaussian with radius 0 reads its radius and returns: no gradient buffer, no
  parameter and no moment is touched (sparse Adam, as in `SelectiveAdam`);
- a visible Gaussian computes its mean, raw-quaternion and log-scale gradients
  with the same vector-Jacobian products as `fully_fused_projection` backward
  (pinhole, no covariance input, no compensation; the log-scale chain rule
  `v_log_scale = v_scale * scale` in the kernel) and applies the `SelectiveAdam`
  update (no bias correction) in registers: no gradient tensors, no atomics
  (one camera), one pass over parameters and moments;
- moments are float32 or bfloat16 (PennyGS's compact moments); arithmetic is
  float32.

It updates its inputs in place and bypasses autograd, so it cannot include
gradients that other losses put on the same parameters, and it computes no
camera gradients. It is a stage replacement for a training loop that owns the
whole update, not a drop-in optimizer.

## Results

Stage "projection backward, then update means / quaternions / log-scales" (ms):

| View | Visible | proj bwd + dense `torch.optim.Adam` (fused, fp32) | proj bwd + `SelectiveAdam` | fused, fp32 moments | fused, bf16 moments |
|---|---:|---:|---:|---:|---:|
| 0 | 260,011 | 2.171 | 0.809 (2.68x) | 0.613 (3.54x) | 0.441 (4.92x) |
| 250 | 2,527,193 | 2.471 | 1.718 (1.44x) | 0.982 (2.52x) | 0.715 (3.46x) |
| 500 | 320,001 | 2.151 | 0.880 (2.44x) | 0.716 (3.01x) | 0.504 (4.27x) |
| 750 | 689,503 | 2.232 | 1.112 (2.01x) | 1.003 (2.23x) | 0.649 (3.44x) |
| 1000 | 145,831 | 2.099 | 0.676 (3.11x) | 0.416 (5.05x) | 0.309 (6.80x) |
| 1250 | 146,389 | 2.090 | 0.667 (3.13x) | 0.409 (5.11x) | 0.305 (6.85x) |
| **geomean** | | 1.00x | **2.39x** | **3.40x** | **4.76x** |

Accuracy of the cumulative update of 10 steps from identical states (relative
L2, max over the three parameters): fused fp32 vs `SelectiveAdam` 3.6e-06 to
1.6e-03 (Adam's first steps are sign-sensitive for near-zero gradients, which the
two evaluation orders round differently); bf16 vs fp32 moments 1.5e-03 on every
view. Dense Adam also decays the moments of unseen Gaussians, which the sparse
variants do not: a different optimizer, not only a faster one.

Measurements: `measurements/fused_adam_stage.json`.

## Reproduce

```bash
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
python -m pytest -q tests/test_fused_projection_adam.py
PYTHONPATH=$PWD python benchmarks/pennygs/fused_adam_benchmark.py --ply <point_cloud.ply> \
  --cameras <cameras.json> --output stage.json
```
