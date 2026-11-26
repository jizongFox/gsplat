"""PennyGS (arXiv 2611.27525) Sec. 3.3: 32-bit sort keys and single-pass binning,
alone and combined, on a real scene. Speed experiment only.

Variants, all in one process on one build, alternating inside each repeat:

- ``default``: 64-bit ids, two passes;
- ``short_keys``: 32-bit keys, two passes (jizongFox/gsplat#23);
- ``single_pass``: 64-bit ids, one pass (jizongFox/gsplat#24);
- ``combined``: 32-bit keys, one pass, equal-key runs put back in Gaussian index
  order (identical to ``short_keys``);
- ``combined_no_ties``: as ``combined`` without that fix-up (same pairs, order
  inside a run of equal keys depends on scheduling).

Scopes:

- ``binning``: `isect_tiles` + `isect_offset_encode` on fixed projected Gaussians
  (CUDA events). Also reports the runs of equal keys for 64-bit ids and 32-bit
  keys (the fix-up's work);
- ``render``: the full RGB+ED + normals renderer of the parent application
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`) forward and
  backward.
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

# name -> (isect_tiles kwargs, rasterization kwargs); CAPACITY is set per view from
# the default run: 1.25x its intersection count, as a "peak so far" buffer.
CAPACITY = {"value": 0}
VARIANTS = {
    "default": (lambda: {}, lambda: {}),
    "short_keys": (
        lambda: {"short_keys": True},
        lambda: {"isect_short_keys": True},
    ),
    "single_pass": (
        lambda: {"single_pass_capacity": CAPACITY["value"]},
        lambda: {"isect_capacity": CAPACITY["value"]},
    ),
    "combined": (
        lambda: {"short_keys": True, "single_pass_capacity": CAPACITY["value"]},
        lambda: {"isect_short_keys": True, "isect_capacity": CAPACITY["value"]},
    ),
    "combined_no_ties": (
        lambda: {
            "short_keys": True,
            "single_pass_capacity": CAPACITY["value"],
            "single_pass_restore_ties": False,
        },
        lambda: {
            "isect_short_keys": True,
            "isect_capacity": CAPACITY["value"],
            "isect_restore_ties": False,
        },
    ),
}
# reference each variant's outputs are compared against
EXACT_REFERENCE = {
    "short_keys": None,
    "single_pass": "default",
    "combined": "short_keys",
    "combined_no_ties": None,
}


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


def run_stats(sorted_keys):
    """Runs of two or more equal keys in a sorted key array."""
    keys = sorted_keys.to(torch.int64)
    _, counts = torch.unique_consecutive(keys, return_counts=True)
    runs = counts[counts > 1]
    return {
        "runs": int(runs.numel()),
        "pairs_in_runs": int(runs.sum()) if runs.numel() else 0,
        "share_in_runs": float(runs.sum()) / keys.numel() if runs.numel() else 0.0,
        "mean_run": float(runs.double().mean()) if runs.numel() else 0.0,
        "p99_run": (
            float(torch.quantile(runs.double(), 0.99)) if runs.numel() else 0.0
        ),
        "max_run": int(runs.max()) if runs.numel() else 0,
        # insertion-sort work bound of the fix-up: sum of run_length^2
        "sum_run_sq": int((runs.to(torch.int64) ** 2).sum()) if runs.numel() else 0,
    }


def canonical_pairs(ids, flatten_ids):
    order = torch.argsort(flatten_ids.to(torch.int64), stable=True)
    order = order[torch.argsort(ids[order].to(torch.int64), stable=True)]
    return ids[order], flatten_ids[order]


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
        kwargs = VARIANTS[variant][0]()
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
        return start.elapsed_time(end), {
            "offsets": offsets,
            "ids": ids,
            "flatten_ids": flatten_ids,
            "n": flatten_ids.numel(),
        }

    return run


def render_case(inputs, viewmat, K, width, height, generator):
    from gsplat import rasterization

    leaves = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    upstream = None

    def run(variant):
        nonlocal upstream
        raster = functools.partial(rasterization, **VARIANTS[variant][1]())
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


def binning_checks(outputs):
    ref = outputs["default"]
    checks = {}
    for variant, out in outputs.items():
        if variant == "default":
            continue
        check = {
            "intersections": int(out["n"]),
            "offsets_identical_to_default": bool(
                torch.equal(out["offsets"], ref["offsets"])
            ),
        }
        exact = EXACT_REFERENCE[variant]
        if exact is not None:
            check[f"identical_to_{exact}"] = all(
                torch.equal(out[k], outputs[exact][k])
                for k in ("offsets", "ids", "flatten_ids")
            )
        if variant == "combined_no_ties":
            same_keys = torch.equal(out["ids"], outputs["short_keys"]["ids"])
            pairs = canonical_pairs(out["ids"], out["flatten_ids"])
            pairs_ref = canonical_pairs(
                outputs["short_keys"]["ids"], outputs["short_keys"]["flatten_ids"]
            )
            check["same_keys_and_pairs_as_short_keys"] = bool(
                same_keys and all(torch.equal(a, b) for a, b in zip(pairs, pairs_ref))
            )
            check["flatten_ids_identical_to_short_keys"] = bool(
                torch.equal(out["flatten_ids"], outputs["short_keys"]["flatten_ids"])
            )
        checks[variant] = check
    return checks


def render_checks(outputs):
    ref = outputs["default"]
    checks = {}
    for variant, out in outputs.items():
        if variant == "default":
            continue
        rgb = out["image"][..., :3].clamp(0, 1)
        ref_rgb = ref["image"][..., :3].clamp(0, 1)
        mse = float(((rgb - ref_rgb) ** 2).mean())
        check = {
            "rgb_psnr_vs_default": math.inf if mse == 0 else -10 * math.log10(mse),
            "image_max_abs_vs_default": float(
                (out["image"] - ref["image"]).abs().max()
            ),
            "gradient_rel_l2_vs_default": {
                k: rel_l2(g, ref["grads"][k]) for k, g in out["grads"].items()
            },
        }
        for other in ("short_keys",):
            if variant in ("combined", "combined_no_ties"):
                check[f"image_identical_to_{other}"] = bool(
                    torch.equal(out["image"], outputs[other]["image"])
                    and torch.equal(out["alpha"], outputs[other]["alpha"])
                )
                check[f"image_max_abs_vs_{other}"] = float(
                    (out["image"] - outputs[other]["image"]).abs().max()
                )
        checks[variant] = check
    return checks


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
        "warmup": args.warmup,
        "repeats": args.repeats,
        "views": [],
    }
    for view in args.views:
        camera = cameras[view]
        viewmat, K = (t.cuda() for t in rb.camera_tensors(camera))
        width, height = int(camera["width"]), int(camera["height"])
        binning = binning_case(inputs, viewmat, K, width, height)
        if args.scope == "binning":
            run = binning
        else:
            run = render_case(
                inputs,
                viewmat,
                K,
                width,
                height,
                torch.Generator().manual_seed(2026 + view),
            )
        _, first = binning("default")
        CAPACITY["value"] = int(1.25 * first["n"])
        info = {"intersections": int(first["n"]), "capacity": CAPACITY["value"]}
        if args.scope == "binning":
            _, short = binning("short_keys")
            info["equal_key_runs"] = {
                "ids64": run_stats(first["ids"]),
                "keys32": run_stats(short["ids"]),
            }
        times = {v: [] for v in args.variants}
        checks = {}
        for repeat in range(args.warmup + args.repeats):
            outputs = {}
            for variant in args.variants:
                ms, out = run(variant)
                if repeat >= args.warmup:
                    times[variant].append(ms)
                if repeat == args.warmup:
                    outputs[variant] = out
            if outputs:
                checks = (
                    binning_checks(outputs)
                    if args.scope == "binning"
                    else render_checks(outputs)
                )
                del outputs
        if args.scope == "binning":
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
        print(json.dumps(report["views"][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
