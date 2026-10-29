"""Benchmark gsplat's differentiable rasterizer on a 3DGS PLY scene.

By default projection, SH, intersections and rasterization are timed together.
Use --raster-only to time detached screen-space inputs instead. Reference checks
and a single optional profiler pass run separately from timing.
Use --render-mode RGB+ED --absgrad for supplementary production-setting runs.

Select this worktree's Python implementation without installing it, for example:
PYTHONPATH="$PWD/submodules/gsplat" <venv>/bin/python \
    benchmarks/gsplat_blog_benchmark.py --extension /path/to/csrc.so --output run.json
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import platform
import sys
import typing as t
from pathlib import Path

import gsplat
import numpy as np
import torch
from plyfile import PlyData

from gsplat.cuda import _wrapper
from gsplat.rendering import rasterization

DEFAULT_SCENE = None  # Supply --ply and --cameras for an external scene.


def load_scene(ply_path: Path, camera_path: Path, max_points: int, camera_index: int):
    vertex = PlyData.read(str(ply_path), mmap=True)["vertex"]
    n = len(vertex)
    if max_points > 0 and max_points < n:
        indices = np.linspace(0, n - 1, max_points, dtype=np.int64)
    else:
        indices = np.arange(n, dtype=np.int64)

    def column(name):
        return np.asarray(vertex[name])[indices].astype(np.float32, copy=False)

    means = np.stack([column(x) for x in ("x", "y", "z")], axis=1)
    scales = np.stack([column(f"scale_{i}") for i in range(3)], axis=1)
    quats = np.stack([column(f"rot_{i}") for i in range(4)], axis=1)
    opacities = column("opacity")

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    coeffs = np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1)[:, None, :]
    if rest_names:
        rest = np.stack([column(x) for x in rest_names], axis=1)
        rest = rest.reshape(len(indices), 3, -1).transpose(0, 2, 1)
        colors = np.concatenate([coeffs, rest], axis=1)
        degree = int(round(np.sqrt(colors.shape[1])) - 1)
    else:
        colors, degree = coeffs, 0

    with camera_path.open() as f:
        camera = json.load(f)[camera_index]
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float32)
    viewmat = np.linalg.inv(c2w)
    K = np.array(
        [[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]],
        dtype=np.float32,
    )
    device = torch.device("cuda")
    tensors = [torch.from_numpy(x).to(device=device) for x in (means, quats, scales)]
    opacity = torch.from_numpy(opacities).to(device=device)
    colors_t = torch.from_numpy(colors).to(device=device)
    return (
        tensors,
        opacity,
        colors_t,
        torch.from_numpy(viewmat)[None].to(device),
        torch.from_numpy(K)[None].to(device),
        int(camera["width"]),
        int(camera["height"]),
        degree,
        n,
    )


def compare_tensors(
    actual: t.Dict[str, torch.Tensor],
    expected: t.Dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> t.Dict[str, t.Any]:
    if actual.keys() != expected.keys():
        raise ValueError("Reference tensor names differ")
    results = {}
    for name, value in actual.items():
        other = expected[name]
        if value.shape != other.shape:
            raise ValueError(f"Reference shape differs for {name}")
        finite = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
        delta = (value.double() - other.double()).abs()
        results[name] = {
            "finite": finite,
            "max_abs": (
                (float(delta.max()) if delta.numel() else 0.0) if finite else None
            ),
            "rms": (
                (float(delta.square().mean().sqrt()) if delta.numel() else 0.0)
                if finite
                else None
            ),
            "max_relative": (
                float((delta / other.double().abs().clamp_min(1e-12)).max())
                if delta.numel() and finite
                else (0.0 if finite else None)
            ),
            "allclose": finite and torch.allclose(value, other, atol=atol, rtol=rtol),
            "rel_l2": (
                float(delta.norm() / other.double().norm().clamp_min(1e-30))
                if finite
                else None
            ),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
    )
    parser.add_argument("--cameras", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--compact-box", action="store_true")
    parser.add_argument(
        "--compact-box-impl", choices=("sweep", "rect_min"), default="sweep"
    )
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--raster-only", action="store_true")
    parser.add_argument(
        "--render-mode", choices=("RGB", "RGB+D", "RGB+ED"), default="RGB"
    )
    parser.add_argument("--absgrad", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--compare-reference", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--gradient-rel-l2", type=float, default=1e-4)
    parser.add_argument("--forward-max-abs", type=float, default=2e-5)
    args = parser.parse_args()
    if args.ply is None or args.cameras is None:
        parser.error("--ply and --cameras are required for the self-contained gsplat benchmark")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be nonnegative")
    if args.forward_only and (args.save_reference or args.compare_reference):
        parser.error("Reference checks require backward; omit --forward-only")
    if args.timing_only and (args.save_reference or args.compare_reference):
        parser.error("--timing-only cannot save or compare references")

    if args.extension:
        spec = importlib.util.spec_from_file_location(
            "gsplat.csrc", args.extension.resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load extension: {args.extension}")
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        sys.modules["gsplat.csrc"] = extension
        gsplat.csrc = extension

    from gsplat.cuda import _backend

    (
        (means, quats, scales),
        opacity,
        colors,
        viewmat,
        K,
        width,
        height,
        degree,
        total_n,
    ) = load_scene(args.ply, args.cameras, args.max_points, args.camera_index)
    inputs = dict(
        means=means,
        quats=quats,
        log_scales=scales,
        opacity_logits=opacity,
        sh_coefficients=colors,
    )

    def full_render():
        return rasterization(
            means,
            torch.nn.functional.normalize(quats, dim=-1),
            torch.exp(scales),
            torch.sigmoid(opacity),
            colors,
            viewmat,
            K,
            width,
            height,
            sh_degree=degree,
            packed=args.packed,
            render_mode=args.render_mode,
            absgrad=args.absgrad,
            tile_size=args.tile_size,
            compact_box=args.compact_box,
            compact_box_impl=args.compact_box_impl,
        )

    raster_meta = None
    if args.raster_only:
        # The one-time setup render supplies exactly the full path's projection
        # and intersection data. Nothing from this setup is timed or differentiated.
        with torch.no_grad():
            setup_image, setup_alpha, raster_meta = full_render()
            del setup_image, setup_alpha
            camera_ids = raster_meta["camera_ids"]
            gaussian_ids = raster_meta["gaussian_ids"]
            centers = torch.linalg.inv(viewmat)[:, :3, 3]
            if args.packed:
                dirs = means[gaussian_ids] - centers[camera_ids]
                shs = colors[gaussian_ids]
            else:
                dirs = means[None] - centers[:, None]
                shs = colors[None]
            screen_colors = (
                _wrapper.spherical_harmonics(
                    degree, dirs, shs, masks=raster_meta["radii"] > 0
                )
                + 0.5
            ).clamp_min(0.0)
            if args.render_mode in ("RGB+D", "RGB+ED"):
                screen_colors = torch.cat(
                    (screen_colors, raster_meta["depths"][..., None]), dim=-1
                )
            inputs = {
                name: value.detach().contiguous()
                for name, value in {
                    "means2d": raster_meta["means2d"],
                    "conics": raster_meta["conics"],
                    "colors": screen_colors,
                    "opacities": raster_meta["opacities"],
                }.items()
            }

    for x in inputs.values():
        x.requires_grad_(not args.forward_only)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    upstream = tuple(
        torch.randn((1, height, width, channels), generator=generator, device="cuda")
        for channels in (3 if args.render_mode == "RGB" else 4, 1)
    )

    def clear_grads():
        for x in inputs.values():
            x.grad = None

    def run(profile: bool = False):
        def stage(name):
            return (
                torch.profiler.record_function(name)
                if profile
                else contextlib.nullcontext()
            )

        with stage("benchmark.forward"):
            if args.raster_only:
                rendered, alpha = _wrapper.rasterize_to_pixels(
                    inputs["means2d"],
                    inputs["conics"],
                    inputs["colors"],
                    inputs["opacities"],
                    width,
                    height,
                    args.tile_size,
                    raster_meta["isect_offsets"],
                    raster_meta["flatten_ids"],
                    packed=args.packed,
                    absgrad=args.absgrad,
                )
                if args.render_mode == "RGB+ED":
                    rendered = torch.cat(
                        [
                            rendered[..., :-1],
                            rendered[..., -1:] / alpha.clamp(min=1e-10),
                        ],
                        dim=-1,
                    )
                meta = raster_meta
            else:
                rendered, alpha, meta = full_render()
        if not args.forward_only:
            with stage("benchmark.backward"):
                torch.autograd.backward((rendered, alpha), upstream)
        return rendered, alpha, meta

    def snapshot():
        clear_grads()
        image, alpha, meta = run()
        values = {"image": image.detach().cpu(), "alpha": alpha.detach().cpu()}
        if not args.forward_only:
            for name, value in inputs.items():
                if value.grad is None:
                    raise RuntimeError(f"Missing gradient: {name}")
                values[f"grad.{name}"] = value.grad.detach().cpu()
            if args.absgrad:
                means2d = inputs["means2d"] if args.raster_only else meta["means2d"]
                absgrad = getattr(means2d, "absgrad", None)
                if absgrad is None:
                    raise RuntimeError("Missing absolute gradient: means2d")
                values["grad.means2d_abs"] = absgrad.detach().cpu()
        return values

    signature = {
        "ply": str(args.ply.resolve()),
        "cameras": str(args.cameras.resolve()),
        "camera_index": args.camera_index,
        "points_used": means.shape[0],
        "resolution": [width, height],
        "degree": degree,
        "seed": args.seed,
        "raster_only": args.raster_only,
        "packed": args.packed,
        "render_mode": args.render_mode,
        "absgrad": args.absgrad,
        "upstream": "seeded_cuda_randn_image_and_alpha_v1",
        "quaternion_input": "normalized_like_product_wrapper",
    }
    validation = {} if args.timing_only else snapshot()
    finite = {name: bool(torch.isfinite(x).all()) for name, x in validation.items()}
    checks = {"finite": finite}
    passed = all(finite.values())
    if args.save_reference or args.compare_reference:
        repeat = snapshot()
        checks["baseline_noise"] = compare_tensors(
            repeat, validation, args.atol, args.rtol
        )
        passed = passed and all(x["finite"] for x in checks["baseline_noise"].values())
        if args.save_reference:
            args.save_reference.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "signature": signature,
                    "tensors": validation,
                    "baseline_repeat": repeat,
                    "baseline_noise": checks["baseline_noise"],
                },
                args.save_reference,
            )
        if args.compare_reference:
            reference = torch.load(
                args.compare_reference, map_location="cpu", weights_only=True
            )
            if reference["signature"] != signature:
                raise ValueError("Reference scene/mode/seed signature differs")
            checks["reference_baseline_noise"] = reference["baseline_noise"]
            checks["reference"] = compare_tensors(
                validation, reference["tensors"], args.atol, args.rtol
            )
            for name, result in checks["reference"].items():
                result["alignment_passed"] = result["finite"] and (
                    result["rel_l2"] <= args.gradient_rel_l2
                    if name.startswith("grad.")
                    else result["max_abs"] <= args.forward_max_abs
                )
            passed = passed and all(
                x["alignment_passed"] for x in checks["reference"].values()
            )
        del repeat
    del validation
    clear_grads()
    torch.cuda.empty_cache()
    for _ in range(args.warmup):
        run()
        clear_grads()
    torch.cuda.synchronize()
    times = []
    peak = 0
    output = None
    for _ in range(args.repeats):
        # Drop outputs AND their metadata graph before constructing another graph.
        output = None
        clear_grads()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        peak = max(peak, torch.cuda.max_memory_allocated())
    image, alpha, meta = output
    checks["timed_final_finite"] = {
        "image": bool(torch.isfinite(image).all()),
        "alpha": bool(torch.isfinite(alpha).all()),
    }
    if not args.forward_only:
        checks["timed_final_finite"].update(
            {
                f"grad.{name}": value.grad is not None
                and bool(torch.isfinite(value.grad).all())
                for name, value in inputs.items()
            }
        )
    passed = passed and all(checks["timed_final_finite"].values())
    summary = {
        "image_mean": (
            float(image.detach().mean()) if torch.isfinite(image).all() else None
        ),
        "alpha_mean": (
            float(alpha.detach().mean()) if torch.isfinite(alpha).all() else None
        ),
        "isects": int(meta["flatten_ids"].numel()),
        "max_flatten_id": (
            int(meta["flatten_ids"].max()) if meta["flatten_ids"].numel() else None
        ),
    }
    del image, alpha, meta, output
    clear_grads()

    profile_summary = None
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            profiled_output = run(profile=True)
            torch.cuda.synchronize()
            del profiled_output
        profile_summary = [
            {
                "name": event.key,
                "calls": event.count,
                "device_total_ms": event.device_time_total / 1000,
                "self_device_ms": event.self_device_time_total / 1000,
            }
            for event in profiler.key_averages()
            if event.device_time_total > 0
        ]
        clear_grads()

    result = {
        **signature,
        "points_total": total_n,
        "points_used": means.shape[0],
        "degree": degree,
        "resolution": [width, height],
        "forward_only": args.forward_only,
        "packed": args.packed,
        "compact_box": args.compact_box,
        "compact_box_impl": args.compact_box_impl,
        "tile_size": args.tile_size,
        "atol": args.atol,
        "rtol": args.rtol,
        "gradient_rel_l2": args.gradient_rel_l2,
        "forward_max_abs": args.forward_max_abs,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_only": args.timing_only,
        "checks": checks,
        "passed": passed,
        "profile": profile_summary,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gsplat_version": getattr(gsplat, "__version__", None),
            "gsplat_path": gsplat.__file__,
            "wrapper_path": _wrapper.__file__,
            "backend_path": getattr(_backend._C, "__file__", None),
            "backend_sha256": hashlib.sha256(
                Path(_backend._C.__file__).read_bytes()
            ).hexdigest(),
        },
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "samples_ms": times,
        "peak_memory_gib": peak / 2**30,
        **summary,
    }
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not passed:
        raise SystemExit("Non-finite values or reference comparison failure")


if __name__ == "__main__":
    main()
