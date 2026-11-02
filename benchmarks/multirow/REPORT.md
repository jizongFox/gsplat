# S06 2DGS multi-row / pipe-balancing experiment

## Decision

This is a documented negative experiment. No CUDA kernel change is retained:
the proposed multi-row execution cannot preserve the current 2DGS backward
contract, including exact densification behavior, without introducing a new
ordering or synchronization protocol.

The experiment is based on fresh `dev/dc` commit `9b331bc` in the isolated
`experiment/2dgs-s06-multirow` worktree. The parent checkout and other
worktrees were not modified.

## Candidate that was evaluated

The candidate was to split a normal 16x16 tile into independent row groups and
pipeline Gaussian batches so one group could evaluate pixels while another
group loaded the next batch. The intent was to reduce idle time caused by
block-wide `block.sync()` calls in
`gsplat/cuda/csrc/rasterize_to_pixels_2dgs_{fwd,bwd}.cu`.

## Safety analysis

### Forward

Forward rendering is independently pixel-local, but all rows in a tile share
the same staged Gaussian batch. A row-group implementation has only two safe
choices:

1. Give every group a private staging slice. This multiplies Gaussian global
   loads and shared-memory use, and is a different warp-private experiment,
   not a pipe-balancing optimization.
2. Reuse one staging slice between groups. This requires a block-wide
   producer/consumer protocol; replacing `block.sync()` with row-local
   synchronization allows a consumer to observe partially written
   `id_batch`, `xy_opacity_batch`, or transform rows.

The forward `T`, termination, median, and distortion state is per pixel, so it
does not provide a safe cross-row reduction to amortize this protocol.

### Backward

Backward is stricter than forward. Each pixel traverses the same Gaussian list
back-to-front while carrying `T`, color/normal buffers, distortion state, and
median/depth state. These values are recurrence state and cannot be pipelined
across rows without changing the mathematical traversal.

Each primitive then performs warp reductions followed by global gradient
atomics. A row-group split changes the participating lane set and atomic
arrival order; it therefore cannot promise exact parity, even if the reduced
values are mathematically equivalent.

### Densification blocker

The current backward kernel writes the densification proxy directly:

```cpp
v_densify_ptr[0] = v_ray_transforms_ptr[2] * depth;
v_densify_ptr[1] = v_ray_transforms_ptr[5] * depth;
```

This is deliberately not an atomic accumulation. A multi-row schedule would
change which pixel's write is observed when a Gaussian contributes to several
rows, so it would change densification output even if image and ordinary
gradient tolerances passed. Preserving this behavior requires retaining the
current execution/order contract or adding an explicit deterministic staging
and selection mechanism, which is outside an isolated S06 optimization.

## Resource and test results

| Item | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, SM 8.9 |
| CUDA compiler | `/usr/local/cuda/bin/nvcc`, CUDA 12.8 toolchain |
| GPU memory at inspection | 24,564 MiB total; 1,044 MiB used |
| Required alias | `@venv/bin/python` is absent |
| Approved existing environment | `/home/jizong/workspace/dConstruct/gaussian-splatting/venv/bin/python`, PyTorch 2.10.0+cu128; used for the separate S05 experiment |
| Candidate build | Not run; no candidate source was retained |
| Parity | Not claimed; no behavior was changed |
| Timing | Not claimed; there is no candidate binary |

No timing or parity number is reported because benchmarking a hypothetical
kernel would not provide evidence for an implementation. The existing
2DGS regression and densification checks remain the appropriate follow-up if a
future design introduces an explicit deterministic multi-row schedule.

## Conclusion

S06 is rejected as unsafe in its requested isolated form. The limiting issue
is not merely register pressure: shared staging needs a correct producer/
consumer barrier, backward needs per-pixel recurrence isolation, and
densification has an intentional direct-write ordering contract. A future
optimization would need a separately specified deterministic densify reduction
and a full sanitizer/parity matrix before it can be considered.

This is a **source-level negative design evaluation**, not a measured performance
result. No CUDA code or 2DGS behavior was changed, so there is no candidate
binary to compare. This rejects the naïve row-group/barrier-removal design, not
every possible S06-inspired implementation.
