"""Predeclared mean-gradient comparison, not a retry-until-pass experiment.

Run baseline and candidate in separate processes with the same arguments and
different --extension paths. Save large references outside Git, e.g.
/tmp/opencode/baseline-rgb-mean.pt. Use PYTHONPATH=submodules/gsplat and the
provided Python environment. Default: camera 0, all points, 16 backward passes
for both baseline and candidate, in either mode.

This estimates alignment of arithmetic mean gradients under CUDA accumulation
noise. It does NOT overturn historical single-shot failures, establish per-pass
alignment, or provide a confidence interval. No warmups, adaptive stopping,
discarded passes, or automatic retries are performed.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import typing as t
from pathlib import Path

import gsplat
import torch

from gsplat_blog_benchmark import load_scene

GRADIENT_REL_L2 = 1e-4
FORWARD_MAX_ABS = 2e-5
SEED = 2026


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def accumulate(
    means: t.Dict[str, torch.Tensor],
    name: str,
    value: torch.Tensor,
    repeats: int,
) -> bool:
    """Keep one mean and one transient CPU tensor, never all pass snapshots."""
    sample = value.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(sample).all())
    if name not in means:
        means[name] = torch.zeros_like(sample)
    means[name].add_(sample.div_(repeats))
    return finite


def compare(
    actual: t.Dict[str, torch.Tensor], expected: t.Dict[str, torch.Tensor]
) -> t.Dict[str, t.Any]:
    if actual.keys() != expected.keys():
        raise ValueError("Reference tensor names differ")
    metrics = {}
    for name, value in actual.items():
        reference = expected[name]
        if reference.shape != value.shape or reference.dtype != torch.float64:
            raise ValueError(f"Reference shape/dtype differs: {name}")
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        delta = value - reference
        max_abs = float(delta.abs().max()) if finite and delta.numel() else 0.0
        rel_l2 = (
            float(delta.norm() / reference.norm().clamp_min(1e-30)) if finite else None
        )
        passed = finite and (
            rel_l2 <= GRADIENT_REL_L2
            if name.startswith("grad.")
            else max_abs <= FORWARD_MAX_ABS
        )
        metrics[name] = {
            "finite": finite,
            "max_abs": max_abs if finite else None,
            "rel_l2": rel_l2,
            "exact_equal": bool(torch.equal(value, reference)),
            "passed": bool(passed),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--mode", choices=("RGB", "RGB+ED"), default="RGB")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
    )
    parser.add_argument("--cameras", type=Path, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0, help="0 means all points")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--save-reference", type=Path)
    action.add_argument("--compare-reference", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    if args.ply is None or args.cameras is None:
        parser.error("--ply and --cameras are required for the self-contained gsplat benchmark")
    if args.repeats < 1 or args.max_points < 0 or args.camera_index < 0:
        parser.error("repeats must be positive; max-points/camera-index nonnegative")
    if args.save_reference and args.save_reference.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("Save large PT references outside Git, e.g. /tmp/opencode/")

    torch.set_num_threads(8)
    # Match the blog harness: install gsplat.csrc before importing _backend.
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
    from gsplat.rendering import rasterization

    if _backend._C is not extension:
        raise RuntimeError("Backend did not select the requested extension")
    extension_identity = {
        "path": str(args.extension.resolve()),
        "sha256": file_hash(args.extension),
    }
    (
        (means, quats, log_scales),
        opacity_logits,
        sh,
        viewmat,
        K,
        width,
        height,
        degree,
        total_n,
    ) = load_scene(args.ply, args.cameras, args.max_points, args.camera_index)
    inputs = {
        "means": means,
        "quats": quats,
        "log_scales": log_scales,
        "opacity_logits": opacity_logits,
        "sh_coefficients": sh,
    }
    for value in inputs.values():
        value.requires_grad_(True)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    upstream = tuple(
        torch.randn((1, height, width, channels), generator=generator, device="cuda")
        for channels in (3 if args.mode == "RGB" else 4, 1)
    )
    signature = {
        "protocol": "predeclared_raw_parameter_mean_v1",
        "script_sha256": file_hash(Path(__file__)),
        "ply": str(args.ply.resolve()),
        "ply_sha256": file_hash(args.ply),
        "cameras": str(args.cameras.resolve()),
        "cameras_sha256": file_hash(args.cameras),
        "camera_index": args.camera_index,
        "points_used": int(means.shape[0]),
        "points_total": total_n,
        "resolution": [width, height],
        "degree": degree,
        "mode": args.mode,
        "absgrad": args.mode == "RGB+ED",
        "packed": False,
        "tile_size": 16,
        "compact_box": True,
        "compact_box_impl": "sweep",
        "settings_note": (
            "Compact-box sweep corrected before candidate ensemble collection; "
            "unpaired baseline-only results replaced. Count and thresholds unchanged."
        ),
        "seed": SEED,
        "upstream": "seeded_cuda_randn_image_and_alpha_v1_fixed_across_passes",
        "upstream_sha256": [tensor_hash(x.cpu()) for x in upstream],
        "repeats": args.repeats,
        "accumulation": "cpu_float64_add_sample_div_repeats_in_order",
        "gradient_rel_l2": GRADIENT_REL_L2,
        "forward_max_abs": FORWARD_MAX_ABS,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }
    reference = None
    if args.compare_reference:
        reference = torch.load(
            args.compare_reference, map_location="cpu", weights_only=True
        )
        if reference["signature"] != signature or reference["repeats"] != args.repeats:
            raise ValueError("Reference protocol/scene/mode/repeats signature differs")

    averages: t.Dict[str, torch.Tensor] = {}
    finite_passes = []
    # Exactly the declared count. Every pass contributes, even if nonfinite.
    for _ in range(args.repeats):
        for value in inputs.values():
            value.grad = None
        image, alpha, meta = rasterization(
            means,
            torch.nn.functional.normalize(quats, dim=-1),
            log_scales.exp(),
            opacity_logits.sigmoid(),
            sh,
            viewmat,
            K,
            width,
            height,
            sh_degree=degree,
            packed=False,
            render_mode=args.mode,
            absgrad=args.mode == "RGB+ED",
            tile_size=16,
            compact_box=True,
            compact_box_impl="sweep",
        )
        torch.autograd.backward((image, alpha), upstream)
        finite = {
            "image": accumulate(averages, "image", image, args.repeats),
            "alpha": accumulate(averages, "alpha", alpha, args.repeats),
        }
        for name, value in inputs.items():
            if value.grad is None:
                raise RuntimeError(f"Missing raw parameter gradient: {name}")
            finite[f"grad.{name}"] = accumulate(
                averages, f"grad.{name}", value.grad, args.repeats
            )
            value.grad = None
        if args.mode == "RGB+ED":
            abs_gradient = getattr(meta["means2d"], "absgrad", None)
            if abs_gradient is None:
                raise RuntimeError("Missing means2d absolute gradient")
            finite["grad.means2d_abs"] = accumulate(
                averages, "grad.means2d_abs", abs_gradient, args.repeats
            )
            del abs_gradient
        finite_passes.append(finite)
        del image, alpha, meta

    metrics = compare(averages, reference["tensors"]) if reference else None
    passed = all(all(entry.values()) for entry in finite_passes)
    if reference:
        passed = (
            passed
            and all(all(entry.values()) for entry in reference["finite_passes"])
            and all(entry["passed"] for entry in metrics.values())
        )
    if args.save_reference:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "signature": signature,
                "extension_identity": extension_identity,
                "repeats": args.repeats,
                "tensors": averages,
                "finite_passes": finite_passes,
            },
            args.save_reference,
        )
    result = {
        "signature": signature,
        "extension_identity": extension_identity,
        "reference_extension_identity": (
            reference["extension_identity"] if reference else None
        ),
        "reference_path": str(args.compare_reference.resolve()) if reference else None,
        "mean_sha256": {name: tensor_hash(x) for name, x in averages.items()},
        "finite_passes": finite_passes,
        "mean_metrics": metrics,
        "passed": bool(passed),
        "comparison_performed": reference is not None,
        "interpretation": (
            "Fixed-count statistical mean-gradient estimate, not a single-shot check. "
            "Historical single-shot failures remain failures of that distinct test; "
            "averaging does not prove per-pass alignment or statistical significance. "
            "No retries, pass selection, or threshold changes."
        ),
    }
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
