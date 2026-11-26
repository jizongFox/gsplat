"""PennyGS (arXiv 2611.27525) forward subtile masks on a real scene.

Scopes, both variants in one process on one build, alternating inside each repeat:

- ``raster``: the forward of `rasterize_to_pixels` alone on fixed projected
  Gaussians and intersections (CUDA events); outputs must be identical;
- ``render``: the full RGB+ED + normals renderer of the parent application
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`), forward and
  backward; images must be identical, gradients are compared (relative L2).
"""

import argparse
import functools
import importlib.util
import json
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "renderer_benchmark", HERE.parent / "reduce_scatter8" / "renderer_benchmark.py"
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
VARIANTS = {"default": False, "subtile": True}


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


def raster_case(inputs, viewmat, K, width, height):
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
    )

    with torch.no_grad():
        scales = inputs["log_scales"].exp()
        quats = rb.quaternion_normalize(inputs["quats"])
        normals = rb.camera_normals(quats, scales, inputs["means"], viewmat)
        rgb = 0.28209479177387814 * inputs["dc"] + 0.5
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
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            16,
            tw,
            th,
            conics=conics,
            opacities=opacities,
            compact_box=True,
        )
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        colors = torch.cat((rgb, normals, depths[0, :, None]), -1)[None]
        background = torch.zeros((1, colors.shape[-1]), device="cuda")

    def run(variant):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.no_grad():
            start.record()
            image, alpha = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                16,
                offsets,
                flatten_ids,
                backgrounds=background,
                subtile_masks=VARIANTS[variant],
            )
            end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), {"image": image, "alpha": alpha}, {}

    return run, int(flatten_ids.numel())


def render_case(inputs, viewmat, K, width, height, generator):
    from gsplat import rasterization

    leaves = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    upstream = None

    def run(variant):
        nonlocal upstream
        raster = functools.partial(
            rasterization, rasterize_subtile_masks=VARIANTS[variant]
        )
        for leaf in leaves.values():
            leaf.grad = None
        torch.cuda.synchronize()
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out, alpha, _, _, _ = rb.render(raster, leaves, viewmat, K, width, height, True)
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
        return (
            (start.elapsed_time(mid), mid.elapsed_time(end)),
            {"image": out.detach(), "alpha": alpha.detach()},
            grads,
        )

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument(
        "--views", type=int, nargs="+", default=[0, 250, 500, 750, 1000, 1250]
    )
    parser.add_argument("--scope", choices=("raster", "render"), default="raster")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
        info = {}
        if args.scope == "raster":
            run, info["intersections"] = raster_case(inputs, viewmat, K, width, height)
        else:
            run = render_case(
                inputs,
                viewmat,
                K,
                width,
                height,
                torch.Generator().manual_seed(2026 + view),
            )
        times = {v: [] for v in VARIANTS}
        check, reference = {}, None
        for repeat in range(args.warmup + args.repeats):
            for variant in VARIANTS:
                ms, out, grads = run(variant)
                if repeat >= args.warmup:
                    times[variant].append(ms)
                if repeat == args.warmup:
                    if variant == "default":
                        reference = (out, grads)
                    else:
                        check = {
                            "outputs_identical": all(
                                torch.equal(out[k], reference[0][k]) for k in out
                            ),
                            "gradient_rel_l2": {
                                k: rel_l2(g, reference[1][k]) for k, g in grads.items()
                            },
                        }
        if args.scope == "raster":
            summary = {v: statistics.median(times[v]) for v in VARIANTS}
        else:
            summary = {
                v: {
                    "forward": statistics.median(t[0] for t in times[v]),
                    "backward": statistics.median(t[1] for t in times[v]),
                    "total": statistics.median(t[0] + t[1] for t in times[v]),
                }
                for v in VARIANTS
            }
        report["views"].append(
            {
                "view": view,
                "image": camera["img_name"],
                **info,
                "median_ms": summary,
                "check": check,
            }
        )
        print(json.dumps(report["views"][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
