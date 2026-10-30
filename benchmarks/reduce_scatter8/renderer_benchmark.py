"""Standalone DC + minor-axis normals RGB+ED benchmark; no parent/PR9 imports."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

PROTOCOL = "dc_minor_axis_rgb_ed_v1"
FORWARD_MAX_ABS = 2e-5  # PR9 acceptance limits, not adjustable upwards.
GRADIENT_REL_L2 = 1e-4


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def quaternion_normalize(quats):
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quats.dtype).eps)


def quaternion_columns(quats):
    w, x, y, z = quaternion_normalize(quats).unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack((
        torch.stack((1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)), -1),
        torch.stack((2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)), -1),
        torch.stack((2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)), -1),
    ), -2)


def camera_normals(quats, scales, means, viewmat):
    """Exact projection.py convention, including post-division +1e-5 and sign(0)."""
    columns = quaternion_columns(quats)
    axis = scales.argmin(-1)
    local = columns.gather(-2, axis[:, None, None].expand(-1, 1, 3)).squeeze(-2)
    rotation = viewmat[:3, :3]
    normal = (rotation[None] * local[:, None, :]).sum(-1)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True) + 1e-5
    normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    camera_mean = (rotation[None] * means[:, None, :]).sum(-1) + viewmat[None, :3, 3]
    return normal * torch.sign((normal * camera_mean).sum(-1))[..., None]


def surface_normals(depth, K):
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=depth.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    uv = torch.stack((x, y, torch.ones_like(x)), -1)
    points = torch.einsum("ij,hwj->hwi", K.inverse(), uv) * depth.permute(1, 2, 0)
    lr = points[1:-1, 2:] - points[1:-1, :-2]
    bt = points[:-2, 1:-1] - points[2:, 1:-1]
    normal = F.normalize(torch.cross(lr, bt, dim=-1), p=2, dim=-1)
    return torch.nan_to_num(-F.pad(normal.permute(2, 0, 1), (1, 1, 1, 1)))


def render(rasterization, inputs, viewmat, K, width, height, compact_box):
    """Full default 3d_normals render path (grad_weight=none, camera coordinates)."""
    scales = inputs["log_scales"].exp()
    quats = quaternion_normalize(inputs["quats"])
    normals = camera_normals(quats, scales, inputs["means"], viewmat)
    rgb = 0.28209479177387814 * inputs["dc"] + 0.5
    colors = torch.cat((rgb, normals), -1)  # 6; ED appends 1; CUDA wrapper pads to 8.
    rendered, alpha, info = rasterization(
        means=inputs["means"], quats=quats, scales=scales,
        opacities=inputs["opacity_logits"].sigmoid(), colors=colors,
        viewmats=viewmat[None], Ks=K[None],
        backgrounds=torch.zeros((1, 6), device=colors.device, dtype=colors.dtype),
        width=width, height=height, packed=False, sh_degree=None,
        render_mode="RGB+ED", absgrad=True, near_plane=0.01, far_plane=1000.0,
        camera_model="pinhole", radius_clip=0.0, compact_box=compact_box,
        compact_box_mult=1.0, compact_box_tau2=None, compact_box_impl="sweep",
    )
    features = rendered[0].permute(2, 0, 1)
    image = features[:3].clamp(0, 1)
    depth = features[6:7]
    normals = features[3:6]
    normals = normals / (torch.linalg.norm(normals, dim=0, keepdim=True) + 1e-6)
    surface = surface_normals(depth, K)
    info["means2d"].retain_grad()
    # Primary upstream is RGB + expected depth (4 channels), plus alpha. Normal
    # postprocessing is executed/timed, but no invented normal loss is introduced.
    return torch.cat((image.permute(1, 2, 0)[None], depth.permute(1, 2, 0)[None]), -1), alpha, info, normals, surface


def load_scene(path, max_points):
    import numpy as np
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    n = len(vertex)
    if n == 0:
        raise ValueError("Empty Gaussian scene")
    ids = np.linspace(0, n - 1, min(n, max_points), dtype=np.int64) if max_points else np.arange(n)

    def columns(names):
        return torch.from_numpy(np.stack([np.asarray(vertex[name])[ids] for name in names], -1).astype(np.float32))

    return {
        "means": columns(("x", "y", "z")),
        "quats": columns([f"rot_{i}" for i in range(4)]),
        "log_scales": columns([f"scale_{i}" for i in range(3)]),
        "opacity_logits": columns(["opacity"]).squeeze(-1),
        "dc": columns([f"f_dc_{i}" for i in range(3)]),
    }, n


def camera_tensors(camera):
    """3DGS JSON rotation is camera-to-world; means remain in WORLD coordinates."""
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(camera["rotation"])
    c2w[:3, 3] = torch.tensor(camera["position"])
    K = torch.tensor([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    return c2w.inverse(), K


def compare(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Reference tensor names differ")
    result = {}
    for name, value in actual.items():
        ref = expected[name]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError(f"Reference shape/dtype differs: {name}")
        finite = bool(torch.isfinite(value).all() and torch.isfinite(ref).all())
        delta = value.double() - ref.double()
        max_abs = float(delta.abs().max()) if finite and delta.numel() else (0.0 if finite else None)
        rel_l2 = float(delta.norm() / ref.double().norm().clamp_min(1e-30)) if finite else None
        passed = finite and (rel_l2 <= GRADIENT_REL_L2 if name.startswith("grad.") else max_abs <= FORWARD_MAX_ABS)
        result[name] = {"finite": finite, "max_abs": max_abs, "rel_l2": rel_l2, "passed": passed}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Read-only Python gsplat checkout")
    parser.add_argument("--extension", type=Path, required=True, help="Prebuilt csrc.so; never JIT-build")
    parser.add_argument("--camera-index", default="all", help="all (default) or one JSON array index")
    parser.add_argument("--compact-box", choices=("on", "off", "both"), default="on")
    parser.add_argument("--max-points", type=int, default=0, help="0 = entire PLY")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save-reference", type=Path)
    group.add_argument("--compare-reference", type=Path)
    args = parser.parse_args(argv)
    if args.max_points < 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("Require max-points/warmup >=0 and repeats >=1")
    if args.camera_index != "all" and (not args.camera_index.isdigit()):
        parser.error("camera-index must be all or a nonnegative integer")
    # No evidence writes inside the selected checkout, including a baseline.
    for path in (args.output, args.save_reference):
        if path and path.resolve().is_relative_to(args.repo.resolve()):
            parser.error("Evidence paths must be outside --repo")
    if args.save_reference and args.save_reference.exists():
        parser.error("Refusing to overwrite an existing baseline reference")
    return args


def signature(args):
    return {
        "protocol": PROTOCOL, "ply_sha256": file_hash(args.ply),
        "cameras_sha256": file_hash(args.cameras), "camera_index": args.camera_index,
        "max_points": args.max_points, "seed": args.seed, "compact_box": args.compact_box,
        "render_settings": {"packed": False, "sh_degree": None, "render_mode": "RGB+ED",
                            "absgrad": True, "near_plane": 0.01, "far_plane": 1000.0,
                            "radius_clip": 0.0, "camera_model": "pinhole", "tile_size": 16,
                            "compact_box_impl": "sweep", "compact_box_mult": 1.0,
                            "compact_box_tau2": None, "background": [0.0] * 6,
                            "clip_image": True, "grad_weight": "none", "normals_coordinate": "camera"},
        "channels": {"colors": 6, "raster_logical": 7, "raster_physical": 8, "scatter_components": 16},
        "upstream": "cpu_randn_seed_plus_camera_index_rgb_ed4_alpha1_v1",
        "thresholds": {"forward_max_abs": FORWARD_MAX_ABS, "gradient_rel_l2": GRADIENT_REL_L2},
        "harness_sha256": file_hash(__file__),
    }


def validate_reference(reference, expected):
    if reference.get("signature") != expected:
        raise ValueError("Reference content hashes or rendering settings differ")
    if not reference.get("passed"):
        raise ValueError("Baseline reference did not pass its own repeat/finite checks")
    for case in reference["cases"].values():
        hashes = {name: tensor_hash(value) for name, value in case["tensors"].items()}
        if hashes != case["tensor_sha256"]:
            raise ValueError("Reference tensor content hashes differ")


def main():
    args = parse_args()
    sig = signature(args)
    reference = None
    if args.compare_reference:
        reference = torch.load(args.compare_reference, map_location="cpu", weights_only=True)
        validate_reference(reference, sig)  # Before importing/loading CUDA.
    cpu_inputs, total = load_scene(args.ply, args.max_points)
    cameras = json.loads(args.cameras.read_text())
    indices = list(range(len(cameras))) if args.camera_index == "all" else [int(args.camera_index)]
    if not indices or max(indices) >= len(cameras):
        raise ValueError("No cameras or camera index out of range")
    compact_options = [True, False] if args.compact_box == "both" else [args.compact_box == "on"]
    expected_keys = {f"camera_{i}/compact_{int(c)}" for i in indices for c in compact_options}
    if reference is not None and reference["cases"].keys() != expected_keys:
        raise ValueError("Reference camera/compact cases differ")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo.resolve()))
    import gsplat

    if not Path(gsplat.__file__).resolve().is_relative_to(args.repo.resolve()):
        raise RuntimeError("gsplat resolved outside selected checkout")
    spec = importlib.util.spec_from_file_location("gsplat.csrc", args.extension.resolve())
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat.csrc"] = extension
    gsplat.csrc = extension
    from gsplat.cuda import _backend
    if Path(_backend._C.__file__).resolve() != args.extension.resolve():
        raise RuntimeError("Backend did not load selected extension")

    inputs = {name: value.cuda().requires_grad_() for name, value in cpu_inputs.items()}
    cases, evidence = {}, {}
    passed = True
    for index in indices:
        camera = cameras[index]
        width, height = int(camera["width"]), int(camera["height"])
        if min(width, height) < 3:
            raise ValueError("Surface normals require width/height >=3")
        viewmat, K = (x.cuda() for x in camera_tensors(camera))
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
        upstream_cpu = [torch.randn((1, height, width, c), generator=generator) for c in (4, 1)]
        upstream = tuple(x.cuda() for x in upstream_cpu)
        for compact in compact_options:
            key = f"camera_{index}/compact_{int(compact)}"

            def clear():
                for value in inputs.values():
                    value.grad = None

            def forward():
                return render(gsplat.rasterization, inputs, viewmat, K, width, height, compact)

            def backward(output):
                torch.autograd.backward(output[:2], upstream)

            def snapshot():
                clear()
                output = forward()
                backward(output)
                image, alpha, info, normal, surface = output
                values = {"image_rgb_ed": image.detach().cpu(), "alpha": alpha.detach().cpu(),
                          "render_normals": normal.detach().cpu(), "surface_normals": surface.detach().cpu()}
                for name, value in inputs.items():
                    if value.grad is None:
                        raise RuntimeError(f"Missing gradient: {name}")
                    values[f"grad.{name}"] = value.grad.detach().cpu()
                for name, value in (("means2d", info["means2d"].grad), ("means2d_abs", getattr(info["means2d"], "absgrad", None))):
                    if value is None:
                        raise RuntimeError(f"Missing gradient: {name}")
                    values[f"grad.{name}"] = value.detach().cpu()
                return values

            values = snapshot()
            noise = compare(snapshot(), values)
            alignment = compare(values, reference["cases"][key]["tensors"]) if reference else None
            case_passed = all(x["passed"] for x in noise.values()) and (alignment is None or all(x["passed"] for x in alignment.values()))
            hashes = {name: tensor_hash(value) for name, value in values.items()}
            upstream_hashes = [tensor_hash(x) for x in upstream_cpu]
            if reference and reference["cases"][key]["upstream_sha256"] != upstream_hashes:
                raise ValueError("Upstream hashes differ")
            if args.save_reference:
                evidence[key] = {"tensors": values, "tensor_sha256": hashes, "upstream_sha256": upstream_hashes, "noise": noise}
            del values
            for _ in range(args.warmup):
                clear()
                output = forward()
                backward(output)
                del output
            torch.cuda.synchronize()
            samples = {name: [] for name in ("forward", "backward", "combined")}
            timed_finite = True
            for _ in range(args.repeats):
                clear()
                start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                start.record()
                output = forward()
                mid.record()
                backward(output)
                end.record()
                end.synchronize()
                for name, a, b in (("forward", start, mid), ("backward", mid, end), ("combined", start, end)):
                    samples[name].append(a.elapsed_time(b))
                timed_finite = timed_finite and all(bool(torch.isfinite(x).all()) for x in (output[0], output[1], output[3], output[4]))
                timed_finite = timed_finite and all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in inputs.values())
                del output
            case_passed = case_passed and timed_finite
            cases[key] = {"passed": case_passed, "resolution": [width, height], "noise": noise,
                          "timed_finite": timed_finite,
                          "reference": alignment, "tensor_sha256": hashes, "upstream_sha256": upstream_hashes,
                          "timings_ms": {name: {"median": statistics.median(times), "mean": statistics.mean(times), "samples": times} for name, times in samples.items()}}
            if args.save_reference:
                evidence[key]["timings_ms"] = cases[key]["timings_ms"]
            if reference:
                cases[key]["baseline_over_candidate_speedup"] = {
                    name: reference["cases"][key]["timings_ms"][name]["median"] / max(statistics.median(times), 1e-12)
                    for name, times in samples.items()
                }
            clear()
            passed = passed and case_passed
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "gsplat_path": gsplat.__file__,
                   "extension_path": str(args.extension.resolve()), "extension_sha256": file_hash(args.extension),
                   "CUDA_HOME": os.environ.get("CUDA_HOME"), "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR")}
    result = {"signature": sig, "passed": passed, "cases": cases, "environment": environment,
              "points_total": total, "points_used": len(cpu_inputs["means"]), "warmup": args.warmup, "repeats": args.repeats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.save_reference:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        with args.save_reference.open("xb") as stream:
            torch.save({"signature": sig, "passed": passed, "cases": evidence, "environment": environment}, stream)
    print(json.dumps({"passed": passed, "output": str(args.output), "cases": {
        key: {"passed": value["passed"], "median_ms": {name: timing["median"] for name, timing in value["timings_ms"].items()},
              "baseline_over_candidate_speedup": value.get("baseline_over_candidate_speedup")}
        for key, value in cases.items()
    }}, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
