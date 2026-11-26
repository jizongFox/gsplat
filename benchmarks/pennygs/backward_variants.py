"""PennyGS (arXiv 2611.27525) backward-rasterization variants on a real scene.

All variants run in one process on one build, so each contribution is isolated:
the same projection, intersections, forward pass and upstream gradients feed
`rasterize_to_pixels(..., backward=<variant>)`. Two scopes:

- ``raster``: the backward of `rasterize_to_pixels` alone (CUDA events around
  `torch.autograd.grad`), i.e. the kernel each variant replaces;
- ``render``: the full RGB+ED + normals renderer of the parent application
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`), forward and
  backward, with `rasterization(..., rasterize_backward=<variant>)`.

Gradients are compared with the "pixel" variant under the unchanged PR9 gates
(gradient relative-L2 <= 1e-4); forward outputs must be identical because the
forward kernel is shared. Variants alternate inside every repeat.
"""

import argparse
import functools
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "renderer_benchmark", HERE.parent / "reduce_scatter8" / "renderer_benchmark.py"
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

GRADIENT_REL_L2 = rb.GRADIENT_REL_L2


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def raster_case(gsplat, inputs, viewmat, K, width, height, generator):
    """Projection, binning and forward once; returns a backward closure per variant."""
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
        tile_size = 16
        tw, th = (width + tile_size - 1) // tile_size, (
            height + tile_size - 1
        ) // tile_size
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tw,
            th,
            conics=conics,
            opacities=opacities,
            compact_box=True,
        )
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        colors = torch.cat((rgb, normals, depths[0, :, None]), -1)[None]  # 7 channels
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (means2d, conics, colors, opacities)
    ]
    background = torch.zeros((1, colors.shape[-1]), device="cuda")
    upstream_c = torch.randn(
        (1, height, width, colors.shape[-1]), generator=generator
    ).cuda()
    upstream_a = torch.randn((1, height, width, 1), generator=generator).cuda()

    def run(variant):
        m, cn, c, o = leaves
        image, alpha = rasterize_to_pixels(
            m,
            cn,
            c,
            o,
            width,
            height,
            tile_size,
            offsets,
            flatten_ids,
            backgrounds=background,
            absgrad=True,
            backward=variant,
        )
        ms, grads = timed(
            lambda: torch.autograd.grad(
                (image, alpha), leaves, (upstream_c, upstream_a)
            )
        )
        absgrad = m.absgrad
        return (
            ms,
            {"image": image.detach(), "alpha": alpha.detach()},
            dict(
                zip(("means2d", "conics", "colors", "opacities"), grads),
                absgrad=absgrad,
            ),
        )

    return run, int(flatten_ids.numel()), int((radii > 0).sum())


def render_case(gsplat, inputs, viewmat, K, width, height, generator):
    from gsplat import rasterization

    leaves = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    upstream = None

    def run(variant):
        nonlocal upstream
        raster = functools.partial(rasterization, rasterize_backward=variant)
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
        grads = {f"grad.{k}": v.grad.detach().clone() for k, v in leaves.items()}
        grads["grad.means2d_abs"] = info["means2d"].absgrad.detach().clone()
        return (
            (start.elapsed_time(mid), mid.elapsed_time(end)),
            {
                "image": out.detach(),
                "alpha": alpha.detach(),
            },
            grads,
        )

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[0, 630, 1259])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "pixel",
            "pixel_lean",
            "chunk",
            "chunk_lean",
            "chunk_cull",
            "chunk_lean_cull",
        ],
    )
    parser.add_argument("--scope", choices=("raster", "render"), default="raster")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.variants[0] == "pixel", "the first variant is the reference"

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
        generator = torch.Generator().manual_seed(2026 + view)
        if args.scope == "raster":
            run, n_isects, visible = raster_case(
                gsplat, inputs, viewmat, K, width, height, generator
            )
            info = {"intersections": n_isects, "visible": visible}
        else:
            run = render_case(gsplat, inputs, viewmat, K, width, height, generator)
            info = {}
        times = {v: [] for v in args.variants}
        reference = None
        checks = {}
        for repeat in range(args.warmup + args.repeats):
            for variant in args.variants:
                ms, outputs, grads = run(variant)
                if repeat >= args.warmup:
                    times[variant].append(ms)
                if repeat == args.warmup:
                    if variant == "pixel":
                        reference = (outputs, grads)
                    else:
                        out_ref, grad_ref = reference
                        checks[variant] = {
                            "forward_identical": all(
                                torch.equal(outputs[k], out_ref[k]) for k in outputs
                            ),
                            "gradient_rel_l2": {
                                k: rel_l2(g, grad_ref[k]) for k, g in grads.items()
                            },
                            "finite": all(
                                bool(torch.isfinite(g).all()) for g in grads.values()
                            ),
                        }
                        checks[variant]["passed"] = (
                            checks[variant]["forward_identical"]
                            and checks[variant]["finite"]
                            and max(checks[variant]["gradient_rel_l2"].values())
                            <= GRADIENT_REL_L2
                        )
        if args.scope == "raster":
            summary = {v: statistics.median(times[v]) for v in args.variants}
        else:
            summary = {
                v: {
                    "forward": statistics.median(t[0] for t in times[v]),
                    "backward": statistics.median(t[1] for t in times[v]),
                    "total": statistics.median(t[0] + t[1] for t in times[v]),
                }
                for v in args.variants
            }
        report["views"].append(
            {
                "view": view,
                "image": camera["img_name"],
                **info,
                "median_ms": summary,
                "checks": checks,
            }
        )
        print(json.dumps(report["views"][-1], indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
