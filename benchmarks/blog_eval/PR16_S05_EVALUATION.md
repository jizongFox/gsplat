# S05 SIMT branch-predication check for 2DGS backward

## Question and adaptation

The blog's S05 removes a divergent branch from a per-splat backward loop. In
gsplat 2DGS backward, each pixel lane computes whether it is inside the image
and whether the absolute intersection index is no later than that pixel's
`last_ids` endpoint. The experimental source rewrote:

```cpp
bool valid = inside;
if (batch_end - t > bin_final) {
    valid = 0;
}
```

as the equivalent single predicate:

```cpp
bool valid = inside && (batch_end - t <= bin_final);
```

No derivative, recurrence, synchronization, reduction, or densification
behavior was intentionally changed. The worktree is a fresh checkout of
`dev/dc` at `9b331bc`; S05 and S06 use separate worktrees/branches.

## Build and compiler result

Fresh base and candidate extensions were built with the approved venv
(PyTorch 2.10.0+cu128), CUDA toolkit 12.8, RTX 4090 SM89. For the production
`COLOR_DIM=4` 2DGS backward kernel, `cuobjdump --dump-sass` reports:

| Variant | SASS instruction slots | `BRA` | `BSSY` | `BSYNC` | registers/thread | SASS hash |
|---|---:|---:|---:|---:|---:|---|
| Base `9b331bc` | 1616 | 16 | 7 | 7 | 80 | `1d5c0dac65826a5b4d34e90e48568d86b3c966f4ea2a4686136e235bf46183eb` |
| S05 predicate rewrite | 1616 | 16 | 7 | 7 | 80 | identical |

The compiler generated **byte-identical SASS** for this kernel specialization.
Thus the source rewrite did not remove the machine-level branch/synchronization
instructions on the tested compiler/architecture. It cannot be claimed as a
kernel optimization on this RTX 4090 build.

## Correctness and timing

- The 52 PR16 synthetic differential cases pass for baseline and candidate,
  including boundaries, partial warps, packed inputs and auxiliary 2DGS losses.
- Three serial Latin-square jobs on water2 show essentially no timing effect:
  full combined geometric-mean ratios are **1.000× native** and **1.000× 4K**;
  raster combined is **1.000× native** and **1.002× 4K**. No speedup is claimed.
- Outputs and ordinary parameter gradients pass, but the candidate's
  `grad/densify` fails the existing `1e-4` comparison on all 12 measured
  view/resolution/path cases (relative-L2 range **3.23e-4–5.39e-3**). The base
  reference is one run, so this does not establish whether the predicate
  rewrite changes densification; the existing cross-warp ordering issue needs
  a dedicated baseline repeat control. This is not a passing full-accuracy
  result and no gate was waived.
- A larger repeated timing series was not considered decision-relevant after
  disassembly proved the candidate kernel identical; no production code change
  is proposed.

## Decision

This direct predicate rewrite is a **negative S05 result for gsplat's current
2DGS kernel/compiler target**: the compiler already lowers the original source
to exactly the same SASS. A real S05 port would need a structurally different
kernel (for example, warp-uniform work compaction or a different traversal),
then fresh densification-safe parity and GPU profiling. This check does not
test the blog's more substantial cmask/bitset worklist implementation.

Measured environment/workload: RTX4090/SM89, PyTorch2.10+cu128, all1,932,132
water2 points, COLMAP-derived views0/623/1247, native and4K, full and raster,
3 Latin-square jobs,20 warmups/40 repeats. Median per-job-mean full combined
speedups are **1.000× native / 1.000× 4K**; raster combined **1.000× / 1.002×**.
Timing and densify-failure reports are in the local ignored artifacts under
`/tmp/opencode/gsplat-s05/evidence/`.

The S06 multi-row assessment is separate and is not included in this S05
worktree or result.
