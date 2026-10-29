"""Multi-camera, multi-resolution megapixel benchmark for the blog benchmark.

This executable intentionally imports the CUDA backend only from :func:`main`.
That makes its argument and camera helpers usable by CPU-only regression tests.
Run baseline and candidate as separate processes, selecting each extension with
``--extension``.

Example (use the same Python wrapper via PYTHONPATH for both binaries)::

    PYTHONPATH=/path/to/gsplat python benchmark_megapixels.py \
        --extension /path/to/base/csrc.so --ply scene.ply --cameras cameras.json \
        --label base --revision ffe --save-reference-dir /tmp/opencode/mp-refs \
        --output /tmp/opencode/mp-base.json

Run the candidate separately with --compare-reference-dir pointing at that
directory. Later serial timing repeats may use --skip-validation instead.
No validation failure relaxes the gates or stops timing. Missing/incompatible
reference files are errors, not successful comparisons. Do not run these workers
concurrently on the same GPU.

JSON records have camera_index, resolution, megapixels, scope and timing.
timing.{training_forward,backward,combined,forward_only} each contains samples_ms,
mean_ms and median_ms. Training components share one A/B/C pass. Image MSE/PSNR
cover RGB only; all four channels and alpha retain the 2e-5 max-abs gate.
Exact images use psnr=null (infinity); any plot cap is display-only.
"""

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import platform
import sys
import subprocess
import typing as t
from pathlib import Path

import numpy as np
import torch

DEFAULT_CAMERA_INDICES = [0, 2552, 5103]
DEFAULT_RESOLUTIONS = [
    (640, 360),
    (960, 540),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3200, 1800),
    (3840, 2160),
]
FORWARD_MAX_ABS = 2e-5
GRADIENT_REL_L2 = 1e-4
PROTOCOL = "megapixels_lossless_rows_v2_unclamped_sh"


def parse_resolution(value: str) -> t.Tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` without importing any CUDA code."""
    try:
        width, height = value.lower().split("x")
        result = (int(width), int(height))
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT") from error
    if result[0] <= 0 or result[1] <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return result


def parse_camera_indices(values: t.Sequence[str]) -> t.List[int]:
    result = []
    for value in values:
        for item in value.split(","):
            try:
                index = int(item)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "camera indices must be integers"
                ) from error
            if index < 0:
                raise argparse.ArgumentTypeError("camera indices must be nonnegative")
            result.append(index)
    if not result:
        raise argparse.ArgumentTypeError("at least one camera index is required")
    return result


def camera_matrices(
    camera: t.Dict[str, t.Any], width: int, height: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return the harness view matrix and FoV-preserving scaled intrinsics."""
    native_width, native_height = int(camera["width"]), int(camera["height"])
    if min(native_width, native_height, width, height) <= 0:
        raise ValueError("Camera and target dimensions must be positive")
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(camera["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float32)
    sx, sy = width / native_width, height / native_height
    K = np.array(
        [
            [camera["fx"] * sx, 0, camera["cx"] * sx],
            [0, camera["fy"] * sy, camera["cy"] * sy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return np.linalg.inv(c2w).astype(np.float32), K


def parse_args(argv: t.Optional[t.Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument(
        "--camera-indices", nargs="+", default=[str(x) for x in DEFAULT_CAMERA_INDICES]
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=parse_resolution, default=DEFAULT_RESOLUTIONS
    )
    parser.add_argument("--scope", choices=("full", "raster", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=0, help="0 means all points")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--revision", default="", help="User-supplied binary source revision"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Timing plus finite checks only; no CPU gradient references",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-reference-dir", type=Path)
    parser.add_argument("--compare-reference-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        args.camera_indices = parse_camera_indices(args.camera_indices)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.repeats < 1 or args.max_points < 0:
        parser.error(
            "warmup must be nonnegative, repeats positive, and max-points nonnegative"
        )
    if args.save_reference_dir and args.compare_reference_dir:
        parser.error(
            "--save-reference-dir and --compare-reference-dir are mutually exclusive"
        )
    if args.skip_validation and (args.save_reference_dir or args.compare_reference_dir):
        parser.error("--skip-validation cannot save or compare references")
    if args.tile_size < 1:
        parser.error("--tile-size must be positive")
    for directory in (args.save_reference_dir, args.compare_reference_dir):
        if directory and directory.resolve().is_relative_to(
            Path(__file__).resolve().parents[2]
        ):
            parser.error("Store large PT references outside Git, e.g. /tmp/opencode/")
    return args


def _stats(samples: t.List[float]) -> t.Dict[str, t.Any]:
    return {
        "samples_ms": samples,
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compress_gradient(
    value: torch.Tensor, row_count: t.Optional[int] = None
) -> t.Dict[str, t.Any]:
    """Losslessly drop exact-zero rows; never quantize or sample gradients.

    Scalar opacities use N rows; unpacked screen inputs [1,N,...] and
    absgrad [1,N,2] use N rows too. All references retain the original shape.
    """
    value = value.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise ValueError("Gradient references require float32 inputs")
    rows = _gradient_rows(value, row_count)
    indices = torch.nonzero((rows != 0).any(dim=1), as_tuple=True)[0]
    return {
        "shape": list(value.shape),
        "row_count": rows.shape[0],
        "indices": indices,
        "values": rows[indices],
    }


def _gradient_rows(
    value: torch.Tensor, row_count: t.Optional[int] = None
) -> torch.Tensor:
    # Rendering supplies N explicitly, including opacity [1,N] and SH [N,K,3].
    n = value.shape[0] if row_count is None else row_count
    columns = math.prod(value.shape) // n if n else math.prod(value.shape[1:])
    return value.reshape(n, columns)


def compare_gradient(
    actual: torch.Tensor, reference: t.Dict[str, t.Any]
) -> t.Dict[str, t.Any]:
    """Check *all* values, including newly nonzero rows absent in the reference."""
    actual = actual.detach().cpu()
    if list(actual.shape) != reference["shape"]:
        raise ValueError("Gradient reference shape differs")
    rows = _gradient_rows(actual, reference["row_count"])
    indices, values = reference["indices"], reference["values"]
    if rows.shape[0] != reference["row_count"] or values.shape != (
        indices.numel(),
        rows.shape[1],
    ):
        raise ValueError("Gradient reference row shape differs")
    finite = bool(torch.isfinite(rows).all() and torch.isfinite(values).all())
    if not finite:
        return {
            "finite": False,
            "max_abs": None,
            "rel_l2": None,
            "exact_equal": False,
            "alignment_passed": False,
        }
    delta = rows[indices].double() - values.double()
    mask = torch.ones(rows.shape[0], dtype=torch.bool)
    mask[indices] = False
    outside = rows[mask].double()
    error_norm = math.hypot(float(delta.norm()), float(outside.norm()))
    rel_l2 = error_norm / max(float(values.double().norm()), 1e-30)
    max_abs = max(
        float(delta.abs().max()) if delta.numel() else 0.0,
        float(outside.abs().max()) if outside.numel() else 0.0,
    )
    return {
        "finite": True,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "exact_equal": max_abs == 0,
        "alignment_passed": rel_l2 <= GRADIENT_REL_L2,
    }


def compare_image(
    actual: torch.Tensor, expected: torch.Tensor, rgb_depth: bool = False
) -> t.Dict[str, t.Any]:
    """Chunk float64 CPU arithmetic to avoid three full double image copies."""
    if actual.shape != expected.shape:
        raise ValueError("Image reference shape differs")
    a, b = actual.detach().cpu().reshape(-1), expected.reshape(-1)
    squared_error, max_abs, finite = 0.0, 0.0, True
    for offset in range(0, a.numel(), 1_000_000):
        x, y = (
            a[offset : offset + 1_000_000].double(),
            b[offset : offset + 1_000_000].double(),
        )
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            finite = False
            break
        delta = x - y
        squared_error += float(delta.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    mse = squared_error / max(a.numel(), 1)
    result = {
        "finite": finite,
        "max_abs": max_abs if finite else None,
        "mse": mse if finite else None,
        "psnr": 10.0 * math.log10(1.0 / mse) if finite and mse else None,
        "exact_equal": finite and max_abs == 0,
        "alignment_passed": finite and max_abs <= FORWARD_MAX_ABS,
    }
    if rgb_depth:
        rgb = compare_image(
            actual[..., :3].contiguous(), expected[..., :3].contiguous()
        )
        depth = compare_image(
            actual[..., 3:].contiguous(), expected[..., 3:].contiguous()
        )
        # Max abs and the alignment gate cover ALL four channels. MSE/PSNR
        # deliberately cover RGB only, never metric ED depth.
        result.update(
            mse=rgb["mse"],
            psnr=rgb["psnr"],
            psnr_channels="RGB",
            rgb_max_abs=rgb["max_abs"],
            depth_max_abs=depth["max_abs"],
            data_range=1.0,
        )
    return result


def _load_extension(path: Path):
    if "gsplat.cuda._backend" in sys.modules or "gsplat.csrc" in sys.modules:
        raise RuntimeError(
            "Use a fresh worker process for each binary (enum registrations collide)"
        )
    import gsplat

    spec = importlib.util.spec_from_file_location("gsplat.csrc", path.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load extension: {path}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat.csrc"] = extension
    gsplat.csrc = extension
    from gsplat.cuda import _backend

    if _backend._C is not extension:
        raise RuntimeError("Backend did not select the requested extension")
    return extension


def raster_colors(raw, viewmat, meta, degree, wrapper):
    """Match rendering.rasterization's SH + 0.5 without a color clamp."""
    dirs = raw["means"][None] - torch.linalg.inv(viewmat)[:, None, :3, 3]
    colors = (
        wrapper.spherical_harmonics(
            degree, dirs, raw["sh_coefficients"][None], masks=meta["radii"] > 0
        )
        + 0.5
    )
    return torch.cat((colors, meta["depths"][..., None]), dim=-1)


def _measure(
    run: t.Callable[
        [bool], t.Tuple[torch.Tensor, torch.Tensor, t.Dict[str, torch.Tensor]]
    ],
    inputs: t.Dict[str, torch.Tensor],
    upstream: t.Tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    repeats: int,
    train: bool,
    cuda: t.Any,
    events: t.Sequence[t.Any],
) -> t.Dict[str, t.Any]:
    def invoke():
        _clear(inputs)
        with contextlib.nullcontext() if train else torch.no_grad():
            image, alpha, meta = run(train)
            if train:
                torch.autograd.backward((image, alpha), upstream)
        return image, alpha, meta

    for _ in range(warmup):
        output = invoke()
        del output
    cuda.synchronize()
    samples, forward_samples, backward_samples = [], [], []
    peak = 0
    checks, summary = {}, {}
    for iteration in range(repeats):
        for value in inputs.values():
            value.grad = None
            if hasattr(value, "absgrad"):
                del value.absgrad
        cuda.reset_peak_memory_stats()
        events[0].record()
        if train:
            image, alpha, meta = run(True)
            events[1].record()
            torch.autograd.backward((image, alpha), upstream)
        else:
            with torch.no_grad():
                image, alpha, meta = run(False)
            events[1].record()
        events[2].record()
        events[2].synchronize()
        forward_ms = events[0].elapsed_time(events[1])
        total_ms = events[0].elapsed_time(events[2])
        forward_samples.append(forward_ms)
        samples.append(total_ms)
        backward_samples.append(events[1].elapsed_time(events[2]))
        peak = max(peak, int(cuda.max_memory_allocated()))
        if iteration == repeats - 1:
            # All scalar readbacks/finite tests are outside A/B/C.
            checks = {
                "image": bool(torch.isfinite(image).all()),
                "alpha": bool(torch.isfinite(alpha).all()),
            }
            if train:
                checks.update(
                    {
                        f"grad.{name}": value.grad is not None
                        and bool(torch.isfinite(value.grad).all())
                        for name, value in inputs.items()
                    }
                )
                absgrad = getattr(meta["means2d"], "absgrad", None)
                checks["grad.means2d_abs"] = absgrad is not None and bool(
                    torch.isfinite(absgrad).all()
                )
                del absgrad
            ids = meta["flatten_ids"]
            summary = {
                "isects": ids.numel(),
                "max_flatten_id": int(ids.max()) if ids.numel() else None,
            }
            del ids
        del image, alpha, meta
    for value in inputs.values():
        value.grad = None
        if hasattr(value, "absgrad"):
            del value.absgrad
    return {
        "forward": _stats(forward_samples),
        "backward": _stats(backward_samples),
        "combined": _stats(samples),
        "peak_memory_gib": peak / 2**30,
        "finite_checks": checks,
        **summary,
    }


def _clear(inputs):
    for value in inputs.values():
        value.grad = None
        if hasattr(value, "absgrad"):
            del value.absgrad


def _snapshot(run, inputs, upstream, expected=None, keep_reference=True):
    """One unmeasured backward; only one dense CPU gradient exists at a time."""
    _clear(inputs)
    image, alpha, meta = run(True)
    torch.autograd.backward((image, alpha), upstream)
    tensors = {"image": image.detach().cpu(), "alpha": alpha.detach().cpu()}
    metrics = {
        name: compare_image(
            value,
            expected["tensors"][name] if expected else value,
            rgb_depth=name == "image",
        )
        for name, value in tensors.items()
    }
    gradients = {}
    row_count = (
        inputs["means"].shape[0] if "means" in inputs else inputs["means2d"].shape[-2]
    )
    absgrad = getattr(meta["means2d"], "absgrad", None)
    if absgrad is None:
        raise RuntimeError("Missing means2d absolute gradient")
    for name in [*inputs, "means2d_abs"]:
        value = absgrad if name == "means2d_abs" else inputs[name].grad
        if value is None:
            raise RuntimeError(f"Missing gradient: {name}")
        cpu = value.detach().cpu()
        if keep_reference or expected is None:
            gradients[name] = compress_gradient(cpu, row_count)
        if expected:
            metrics[f"grad.{name}"] = compare_gradient(cpu, expected["gradients"][name])
        else:
            metrics[f"grad.{name}"] = compare_gradient(cpu, gradients[name])
        del cpu
    ids = meta["flatten_ids"]
    summary = {
        "isects": ids.numel(),
        "max_flatten_id": int(ids.max()) if ids.numel() else None,
    }
    del image, alpha, meta, absgrad, value, ids
    _clear(inputs)
    return {"tensors": tensors, "gradients": gradients}, metrics, summary


def _case(
    args,
    raw,
    camera,
    camera_index,
    width,
    height,
    scope,
    degree,
    settings,
    identity,
    rasterization,
    wrapper,
    upstream_cache,
):
    """All per-case GPU objects die on return (including after an OOM unwind)."""
    view_np, K_np = camera_matrices(camera, width, height)
    viewmat = torch.from_numpy(view_np)[None].cuda()
    K = torch.from_numpy(K_np)[None].cuda()

    def full_render(_train=False):
        return rasterization(
            raw["means"],
            torch.nn.functional.normalize(raw["quats"], dim=-1),
            raw["log_scales"].exp(),
            raw["opacity_logits"].sigmoid(),
            raw["sh_coefficients"],
            viewmat,
            K,
            width,
            height,
            sh_degree=degree,
            **settings,
        )

    inputs, run = raw, full_render
    if scope == "raster":
        with torch.no_grad():
            setup_image, setup_alpha, meta = full_render()
            del setup_image, setup_alpha
            colors = raster_colors(raw, viewmat, meta, degree, wrapper)
            inputs = {
                name: value.detach().contiguous().requires_grad_(True)
                for name, value in {
                    "means2d": meta["means2d"],
                    "conics": meta["conics"],
                    "colors": colors,
                    "opacities": meta["opacities"],
                }.items()
            }
            raster_meta = {
                name: meta[name] for name in ("isect_offsets", "flatten_ids")
            }
            del meta, colors

        def raster_render(_train=False):
            image, alpha = wrapper.rasterize_to_pixels(
                inputs["means2d"],
                inputs["conics"],
                inputs["colors"],
                inputs["opacities"],
                width,
                height,
                settings["tile_size"],
                raster_meta["isect_offsets"],
                raster_meta["flatten_ids"],
                packed=False,
                absgrad=True,
            )
            image = torch.cat(
                (image[..., :-1], image[..., -1:] / alpha.clamp_min(1e-10)), dim=-1
            )
            return image, alpha, {**raster_meta, "means2d": inputs["means2d"]}

        run = raster_render
    # Seed depends on the case, not iteration order or selected subset; both scopes
    # and both binaries receive exactly the same nonuniform upstream values.
    seed = (args.seed + camera_index * 1_000_003 + width * 1009 + height) % (2**63 - 1)
    if "values" not in upstream_cache:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        upstream_cache["values"] = tuple(
            torch.randn(
                (1, height, width, channels), generator=generator, device="cuda"
            )
            for channels in (4, 1)
        )
    upstream = upstream_cache["values"]
    signature = {
        **identity,
        **settings,
        "protocol": PROTOCOL,
        "camera_index": camera_index,
        "resolution": [width, height],
        "scope": scope,
        "degree": degree,
        "seed": seed,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "camera": camera,
        "scaled_K": K_np.tolist(),
        "viewmat": view_np.tolist(),
        "upstream": "seeded_cuda_randn_image4_alpha1_v1",
    }
    name = f"cam{camera_index}-{width}x{height}-{scope}-RGB+ED-seed{seed}.pt"
    expected = None
    if args.compare_reference_dir:
        expected = torch.load(
            args.compare_reference_dir / name, map_location="cpu", weights_only=True
        )
        if expected["signature"] != signature:
            raise ValueError(f"Reference signature differs: {name}")
    try:
        # Exactly three events per case, reused for inference and training.
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        # Both scopes get inference and training-forward curves separately.
        forward_only = _measure(
            run, inputs, upstream, args.warmup, args.repeats, False, torch.cuda, events
        )
        training = _measure(
            run, inputs, upstream, args.warmup, args.repeats, True, torch.cuda, events
        )
        parity, noise, baseline_noise, baseline_provenance = None, None, None, None
        cross_scope_forward = None
        if not args.skip_validation:
            first, parity, _ = _snapshot(run, inputs, upstream, expected)
            baseline_noise = expected["repeat_noise"] if expected else None
            baseline_provenance = expected["provenance"] if expected else None
            del expected
            second, noise, _ = _snapshot(
                run, inputs, upstream, first, keep_reference=False
            )
            del second
            reference_dir = args.save_reference_dir or args.compare_reference_dir
            if scope == "raster" and args.scope == "both" and reference_dir:
                full_name = name.replace("-raster-", "-full-")
                full_reference = torch.load(
                    reference_dir / full_name, map_location="cpu", weights_only=True
                )
                if full_reference["signature"] != {**signature, "scope": "full"}:
                    raise ValueError("Full/raster forward reference signatures differ")
                cross_scope_forward = {
                    key: compare_image(
                        first["tensors"][key],
                        full_reference["tensors"][key],
                        rgb_depth=key == "image",
                    )
                    for key in ("image", "alpha")
                }
                del full_reference
                if not all(v["alignment_passed"] for v in cross_scope_forward.values()):
                    raise ValueError(
                        "Raster-only setup does not match the full renderer"
                    )
            if args.save_reference_dir:
                args.save_reference_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        **first,
                        "signature": signature,
                        "repeat_noise": noise,
                        "provenance": {
                            "extension_path": str(args.extension.resolve()),
                            "extension_sha256": args.extension_sha256,
                            "label": args.label,
                            "revision": args.revision,
                        },
                    },
                    args.save_reference_dir / name,
                )
            del first
        return {
            "status": "ok",
            "seed": seed,
            "reference_file": name,
            "scaled_K": K_np.tolist(),
            "viewmat": view_np.tolist(),
            "camera_identifier": camera,
            "timing": {
                "forward_only": forward_only["forward"],
                "training_forward": training["forward"],
                "backward": training["backward"],
                "combined": training["combined"],
            },
            "peak_memory_gib": max(
                forward_only["peak_memory_gib"], training["peak_memory_gib"]
            ),
            "parity": parity if args.compare_reference_dir else None,
            "finite_checks": {
                "forward_only": forward_only["finite_checks"],
                "training": training["finite_checks"],
            },
            "alignment_passed": (
                all(x["alignment_passed"] for x in parity.values())
                if args.compare_reference_dir
                else None
            ),
            "repeat_noise": noise,
            "baseline_repeat_noise": baseline_noise,
            "baseline_provenance": baseline_provenance,
            "cross_scope_forward": cross_scope_forward,
            "isects": training["isects"],
            "max_flatten_id": training["max_flatten_id"],
        }
    finally:
        _clear(inputs)


def _revision(directory):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    extension = _load_extension(args.extension)
    args.extension_sha256 = _sha(args.extension)
    import gsplat
    from gsplat.cuda import _wrapper
    from gsplat.rendering import rasterization

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gsplat_blog_benchmark import inspect_scene_metadata, load_scene

    torch.set_num_threads(8)
    with args.cameras.open() as stream:
        cameras = json.load(stream)
    for index in args.camera_indices:
        if index >= len(cameras):
            raise ValueError(f"camera index {index} is outside the camera file")
    first = args.camera_indices[0]
    metadata = inspect_scene_metadata(args.ply, args.cameras, args.max_points, first)
    (
        (means, quats, log_scales),
        opacity_logits,
        sh,
        initial_view,
        initial_K,
        _,
        _,
        degree,
        total_n,
    ) = load_scene(
        args.ply, args.cameras, args.max_points, first, scene_metadata=metadata
    )
    del initial_view, initial_K, metadata
    raw = {
        "means": means,
        "quats": quats,
        "log_scales": log_scales,
        "opacity_logits": opacity_logits,
        "sh_coefficients": sh,
    }
    for value in raw.values():
        value.requires_grad_(True)
    settings = {
        "packed": False,
        "render_mode": "RGB+ED",
        "absgrad": True,
        "tile_size": args.tile_size,
        "compact_box": True,
        "compact_box_impl": "sweep",
    }
    identity = {
        "ply_sha256": _sha(args.ply),
        "cameras_sha256": _sha(args.cameras),
        "points_total": total_n,
        "points_used": int(means.shape[0]),
    }
    report = {
        "protocol": PROTOCOL,
        "label": args.label,
        "revision_label": args.revision,
        "extension": {
            "path": str(args.extension.resolve()),
            "sha256": args.extension_sha256,
        },
        "inputs": {
            **identity,
            "ply": str(args.ply.resolve()),
            "cameras": str(args.cameras.resolve()),
        },
        "config": {
            **settings,
            "camera_indices": args.camera_indices,
            "resolutions": [list(x) for x in args.resolutions],
            "scope": args.scope,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "max_points": args.max_points,
            "degree": degree,
            "skip_validation": args.skip_validation,
            "forward_max_abs": FORWARD_MAX_ABS,
            "gradient_rel_l2": GRADIENT_REL_L2,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "gsplat_path": gsplat.__file__,
            "wrapper_path": _wrapper.__file__,
            "wrapper_sha256": _sha(Path(_wrapper.__file__)),
            "backend_path": extension.__file__,
            "script_sha256": _sha(Path(__file__)),
            "repository_revision": _revision(Path(__file__).resolve().parents[2]),
            "gsplat_revision": _revision(Path(gsplat.__file__).parent),
        },
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scopes = ("full", "raster") if args.scope == "both" else (args.scope,)
    for camera_index in args.camera_indices:
        for width, height in args.resolutions:
            # Generate once for this camera/resolution, reuse across both scopes.
            upstream_cache = {}
            for scope in scopes:
                point = {
                    "camera_index": camera_index,
                    "resolution": [width, height],
                    "megapixels": width * height / 1_000_000,
                    "scope": scope,
                }
                try:
                    point.update(
                        _case(
                            args,
                            raw,
                            cameras[camera_index],
                            camera_index,
                            width,
                            height,
                            scope,
                            degree,
                            settings,
                            identity,
                            rasterization,
                            _wrapper,
                            upstream_cache,
                        )
                    )
                except torch.cuda.OutOfMemoryError as error:
                    point.update(status="oom", error=str(error))
                finally:
                    # The failed case's stack must unwind before freeing cached blocks.
                    _clear(raw)
                    gc.collect()
                    torch.cuda.empty_cache()
                report["records"].append(point)
                args.output.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
            upstream_cache.clear()
            torch.cuda.empty_cache()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
