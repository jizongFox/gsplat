"""PennyGS (arXiv 2611.27525) tile-binning variants on a real scene.

Scopes, all variants in one process on one build, alternating inside each repeat:

- ``binning``: `isect_tiles` + `isect_offset_encode` on fixed projected Gaussians
  (CUDA events), with the per-tile ranges compared against the default;
- ``render``: the full RGB+ED + normals renderer of the parent application
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`) forward and
  backward, with the variant's `rasterization` options; the image is compared
  against the default (PSNR of the RGB part, max abs) and so are the gradients
  (relative L2), reported rather than gated because the blending order may change.
"""

import argparse
import functools
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "renderer_benchmark", HERE.parent / "reduce_scatter8" / "renderer_benchmark.py"
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

# name -> (isect_tiles kwargs, rasterization kwargs)
VARIANTS = {
    "default": ({}, {}),
    "short_keys": ({"short_keys": True}, {"isect_short_keys": True}),
}


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


def binning_case(inputs, viewmat, K, width, height):
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
    )

    with torch.no_grad():
        scales = inputs["log_scales"].exp()
        quats = rb.quaternion_normalize(inputs["quats"])
        radii, means2d, depths, conics, _ = fully_fused_projection(
            inputs["means"],
            None,
            quats,
            scales,
            viewmat[None],
            K[None],
            width,
            height,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1000.0,
            radius_clip=0.0,
        )
        opacities = inputs["opacity_logits"].sigmoid()[None]
    tw, th = (width + 15) // 16, (height + 15) // 16

    def run(variant):
        kwargs = VARIANTS[variant][0]
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        _, ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            16,
            tw,
            th,
            conics=conics,
            opacities=opacities,
            compact_box=True,
            **kwargs,
        )
        offsets = isect_offset_encode(ids, 1, tw, th)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), {"offsets": offsets, "n": flatten_ids.numel()}

    return run


def render_case(inputs, viewmat, K, width, height, generator):
    from gsplat import rasterization

    leaves = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    upstream = None

    def run(variant):
        nonlocal upstream
        raster = functools.partial(rasterization, **VARIANTS[variant][1])
        for leaf in leaves.values():
            leaf.grad = None
        torch.cuda.synchronize()
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out, alpha, info, _, _ = rb.render(
            raster, leaves, viewmat, K, width, height, True
        )
        mid.record()
        if upstream is None:
            upstream = (
                torch.randn(out.shape, generator=generator).cuda(),
                torch.randn(alpha.shape, generator=generator).cuda(),
            )
        torch.autograd.backward((out, alpha), upstream)
        end.record()
        torch.cuda.synchronize()
        grads = {k: v.grad.detach().clone() for k, v in leaves.items()}
        return (start.elapsed_time(mid), mid.elapsed_time(end)), {
            "image": out.detach(),
            "alpha": alpha.detach(),
            "grads": grads,
        }

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument(
        "--views", type=int, nargs="+", default=[0, 250, 500, 750, 1000, 1250]
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--scope", choices=("binning", "render"), default="binning")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.variants[0] == "default", "the first variant is the reference"

    import gsplat

    scene, n = rb.load_scene(args.ply, 0)
    inputs = {k: v.cuda() for k, v in scene.items()}
    cameras = json.loads(args.cameras.read_text())
    report = {
        "gsplat": str(Path(gsplat.__file__).parent),
        "device": torch.cuda.get_device_name(),
        "gaussians": n,
        "scope": args.scope,
        "views": [],
    }
    for view in args.views:
        camera = cameras[view]
        viewmat, K = (t.cuda() for t in rb.camera_tensors(camera))
        width, height = int(camera["width"]), int(camera["height"])
        if args.scope == "binning":
            run = binning_case(inputs, viewmat, K, width, height)
        else:
            run = render_case(
                inputs,
                viewmat,
                K,
                width,
                height,
                torch.Generator().manual_seed(2026 + view),
            )
        times = {v: [] for v in args.variants}
        checks, reference = {}, None
        for repeat in range(args.warmup + args.repeats):
            for variant in args.variants:
                ms, out = run(variant)
                if repeat >= args.warmup:
                    times[variant].append(ms)
                if repeat != args.warmup:
                    continue
                if variant == "default":
                    reference = out
                elif args.scope == "binning":
                    checks[variant] = {
                        "intersections": int(out["n"]),
                        "offsets_identical": bool(
                            torch.equal(out["offsets"], reference["offsets"])
                        ),
                    }
                else:
                    rgb, ref_rgb = out["image"][..., :3].clamp(0, 1), reference[
                        "image"
                    ][..., :3].clamp(0, 1)
                    mse = float(((rgb - ref_rgb) ** 2).mean())
                    checks[variant] = {
                        "rgb_psnr_vs_default": (
                            math.inf if mse == 0 else -10 * math.log10(mse)
                        ),
                        "image_max_abs": float(
                            (out["image"] - reference["image"]).abs().max()
                        ),
                        "gradient_rel_l2": {
                            k: rel_l2(g, reference["grads"][k])
                            for k, g in out["grads"].items()
                        },
                    }
        if args.scope == "binning":
            summary = {v: statistics.median(times[v]) for v in args.variants}
            info = {"intersections": int(reference["n"])}
        else:
            summary = {
                v: {
                    "forward": statistics.median(t[0] for t in times[v]),
                    "backward": statistics.median(t[1] for t in times[v]),
                    "total": statistics.median(t[0] + t[1] for t in times[v]),
                }
                for v in args.variants
            }
            info = {}
        report["views"].append(
            {
                "view": view,
                "image": camera["img_name"],
                **info,
                "median_ms": summary,
                "checks": checks,
            }
        )
        print(json.dumps(report["views"][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
