# PennyGS (arXiv 2611.27525) Sec. 3.4: colour staging in the subtile forward

PennyGS stages each batch's colours in shared memory next to the other batch
parameters, so a hit reads its colour from shared memory instead of global
memory. This adds it to the subtile forward kernel (`forward="subtile"`) and
measures it for the colours an application actually passes as
`override_color`: depth, RGB, RGB + depth, the parent application's training
renderer (RGB + camera normal + expected depth, 7 channels padded to 8), and
wider feature renders.

## Change

- The thread that loads a batch entry also copies its colours to shared
  memory, channel-major (`[k][entry]`): the stores are conflict-free, and the
  32 pixels of a warp read the same entry, which broadcasts.
- fp32, the blend arithmetic unchanged: outputs are identical to the default
  kernel for any channel meaning (depth would not survive fp16).
- **Only for 1-4 channels** (`subtile_fwd::MAX_STAGE_DIM = 4`, compile time).
  Wider colours are read from global memory per hit, as before; for 5 or more
  channels the kernel is unchanged (8 channels: 47 registers, 11.5 KB).
- No new argument: it is part of `forward="subtile"`.

Cost of staging, per 256-thread block (from `cuobjdump`; no staging / staged):

| Channels | Registers | Static shared memory |
|---:|---|---|
| 1 | 37 / 40 | 11.5 KB / 11.5 KB (the Gaussian ids are no longer needed) |
| 4 | 40 / 45 | 11.5 KB / 14.6 KB |
| 8 | 47 / 48 | 11.5 KB / 18.7 KB |
| 16 | 48 / 48 | 11.5 KB / 26.9 KB |
| 32 | 64 / 79 | 11.5 KB / 43.3 KB |

At 16 channels the staged block fits 3 instead of 5 blocks per SM (shared
memory), at 32 channels 2 instead of 4.

## Measurement

`subtile_staging_benchmark.py` at commit `a622588`, which still has the
experiment variable `GSPLAT_SUBTILE_COLOR_STAGE` to switch the staging mode
inside one process.

- Arms, alternating inside each repeat: `default` kernel; `subtile` (#25,
  colours per hit); `stage_all` (every entry staged); `stage_hit` (only entries
  some warp hits).
- RTX 4090, water2 3DGS model (2,909,851 Gaussians), views 0, 250, 500, 750,
  1000, 1250 at 1280x720 and with intrinsics scaled to 3840x2160.
- 20 warmups, 80 repeats, medians. Each case (view x size x channels) waits for
  an idle GPU and is measured again if another process used the GPU during it
  (the machine is shared): 0 re-measurements in the reported runs.
- 16 and 32 channels: the 7 real channels plus seeded random features. The
  8-channel case passes 7 channels, so every call includes the wrapper's
  padding copy, as in the application.

### Forward raster (`rasterize_to_pixels`)

Geomean over the 6 views; staged = `stage_all`:

| Channels | Size | default (ms) | #25 vs default | staged vs #25 (worst-best view) | staged vs default |
|---:|---|---:|---:|---:|---:|
| 1 | 1280x720 | 0.57 | 1.170x | **1.090x** (1.039-1.143) | 1.276x |
| 1 | 3840x2160 | 1.62 | 1.081x | **1.020x** (1.011-1.032) | 1.102x |
| 3 | 1280x720 | 0.59 | 1.168x | **1.100x** (1.046-1.179) | 1.284x |
| 3 | 3840x2160 | 1.65 | 1.081x | **1.024x** (1.018-1.033) | 1.107x |
| 4 | 1280x720 | 0.60 | 1.157x | **1.107x** (1.046-1.207) | 1.280x |
| 4 | 3840x2160 | 1.66 | 1.072x | **1.018x** (1.013-1.024) | 1.091x |
| 8 | 1280x720 | 1.07 | 1.179x | 1.015x (**0.980**-1.074) | 1.197x |
| 8 | 3840x2160 | 2.10 | 1.101x | 0.997x (**0.964**-1.007) | 1.097x |
| 16 | 1280x720 | 0.86 | 1.195x | 1.090x (**0.966**-1.352) | 1.302x |
| 16 | 3840x2160 | 2.78 | 1.127x | 1.023x (**0.948**-1.064) | 1.153x |
| 32 | 1280x720 | 1.53 | 1.105x | 1.038x (**0.985**-1.179) | 1.147x |
| 32 | 3840x2160 | 10.44 | **0.984x** | 0.988x (**0.960**-0.998) | 0.972x |

- Staging 1-4 channels is faster on every view at both sizes: 1.09-1.11x at
  1280x720, 1.02x at 3840x2160 on top of #25.
- From 8 channels on, some view is slower at some size, so staging is not
  used there. The rule was "never slower on any view or size".
- `stage_hit` is within 0.2 % of `stage_all` (geomean over all cases), so the
  simpler `stage_all` is kept.
- New limit of #25 itself: at 32 channels and 3840x2160, the subtile kernel is
  1.6 % slower than the default kernel. At 8 channels and both sizes it is
  faster.
- All arms' images and alphas are identical to the default kernel in every case.

### Full renderer, 8 channels (forward + backward)

| Size | #25 vs default | staged vs #25 |
|---|---:|---:|
| 1280x720 | 1.0121x | 1.0016x |
| 3840x2160 | 1.0055x | 0.9998x |

Staging is neutral for the 8-channel training renderer, as the raster results
predict; with the 1-4 channel threshold that renderer runs the unchanged #25
kernel. Images are identical; gradients differ within the run-to-run noise of
the backward atomics: the default kernel against itself gives the same
quaternion-gradient relative L2 (up to 7.1e-4 at 3840x2160, view 1250) as the
arms against the default (up to 8.4e-4).

### Confirmation on the final kernel

The same benchmark on the final build (1-4 channels staged, wider read per hit;
the three subtile arms then run the same kernel), forward raster, `subtile` vs
`default`, geomean over the 6 views:

| Channels | 1280x720 (worst-best view) | 3840x2160 (worst-best view) | experiment: staged / unstaged |
|---:|---:|---:|---|
| 1 | 1.277x (1.157-1.582) | 1.102x (1.082-1.126) | 1.276x / 1.170x; 1.102x / 1.081x |
| 3 | 1.284x (1.170-1.567) | 1.107x (1.082-1.129) | 1.284x / 1.168x; 1.107x / 1.081x |
| 4 | 1.273x (1.170-1.538) | 1.091x (1.070-1.107) | 1.280x / 1.157x; 1.091x / 1.072x |
| 8 | 1.174x (1.109-1.345) | 1.100x (1.058-1.176) | 1.197x / 1.179x; 1.097x / 1.101x |

1-4 channels match the staged experiment arm, 8 channels the unstaged one.
Outputs identical; 2 cases were re-measured because another process used the
GPU. Measurements: `measurements/subtile_staging_{raster,render,final}.json`.

## Reproduce

```bash
git checkout a622588
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
PYTHONPATH=$PWD python benchmarks/pennygs/subtile_staging_benchmark.py --ply <point_cloud.ply> \
    --cameras <cameras.json> --scope raster --output raster.json   # then --scope render
```
