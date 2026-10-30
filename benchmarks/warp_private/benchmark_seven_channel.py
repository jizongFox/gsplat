"""Standalone raster benchmark; run each --extension in a fresh, serial process.

Save baseline references with --save-reference-dir, then run the candidate with
--compare-reference-dir. Projection, SH evaluation, local +Z normal conversion,
hashing and validation are outside CUDA-event timing. Seven logical attributes
are RGB, camera Z, and camera-space normal XYZ; the existing wrapper pads to 8.
All attributes are alpha-weighted sums (depth is D, not expected depth ED).
Normals are neither face-forwarded nor renormalized after compositing.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BLOG = ROOT / "benchmarks/blog_eval"
spec = importlib.util.spec_from_file_location("warp_private_pr9", BLOG / "benchmark_megapixels.py")
pr9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr9)

SCENE = Path("/home/jizong/workspace/dConstruct/data/2026-10-10-water2/outputs/56ee2948")
PROTOCOL = "warp_private_rgb_d_local_z_v1"
NORMAL_CONVENTION = "normalized_wxyz_quaternion_local_+Z_to_world_to_camera_no_flip"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ply", type=Path, default=SCENE / "point_cloud/iteration_020000/point_cloud.ply")
    parser.add_argument("--colmap", type=Path, default=SCENE / "exports/colmap/sparse")
    parser.add_argument("--camera-indices", nargs="+", default=["0,623,1247"])
    parser.add_argument("--resolutions", nargs="+", default=["native"], help="native, 4k, or WIDTHxHEIGHT")
    parser.add_argument("--channels", type=int, choices=(3, 4, 7), default=7)
    parser.add_argument("--tile-size", type=int, choices=(16,), default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", default="")
    parser.add_argument("--revision", default="", help="Source revision of the selected binary")
    parser.add_argument("--output", type=Path, required=True)
    refs = parser.add_mutually_exclusive_group()
    refs.add_argument("--save-reference-dir", type=Path)
    refs.add_argument("--compare-reference-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        args.camera_indices = pr9.parse_camera_indices(args.camera_indices)
        args.resolutions = [x if x in ("native", "4k") else pr9.parse_resolution(x) for x in args.resolutions]
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    for directory in (args.save_reference_dir, args.compare_reference_dir):
        if directory and directory.resolve().is_relative_to(ROOT):
            parser.error("Store references outside the repository")
    return args


def quaternion_matrix(q):
    """COLMAP/3DGS scalar-first quaternion, normalized before use."""
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def colmap_cameras(directory):
    """Convert undistorted COLMAP text to PR9 c2w JSON, sorted by image name.

    Consume the second image line even when empty; it contains observations,
    not another pose. Reject distorted models rather than silently dropping them.
    """
    intrinsics = {}
    for line in (directory / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cid, model, width, height, *params = line.split()
        p = list(map(float, params))
        if model == "PINHOLE" and len(p) == 4:
            fx, fy, cx, cy = p
        elif model == "SIMPLE_PINHOLE" and len(p) == 3:
            fx, cx, cy = p
            fy = fx
        else:
            raise ValueError(f"Unsupported camera model: {model}; export undistorted pinhole cameras")
        if min(int(width), int(height), fx, fy) <= 0 or not np.isfinite(p).all():
            raise ValueError("Invalid camera intrinsics")
        if int(cid) in intrinsics:
            raise ValueError("Duplicate COLMAP camera ID")
        intrinsics[int(cid)] = dict(width=int(width), height=int(height), fx=fx, fy=fy, cx=cx, cy=cy)
    lines = iter((directory / "images.txt").read_text().splitlines())
    cameras = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError("Invalid COLMAP image pose")
        iid, *pose, cid, name = fields
        rotation = quaternion_matrix(list(map(float, pose[:4])))
        translation = np.asarray(pose[4:], dtype=np.float64)
        if not np.isfinite(translation).all():
            raise ValueError("Invalid camera translation")
        cameras.append({**intrinsics[int(cid)], "id": int(iid), "img_name": name,
                        "rotation": rotation.T.tolist(), "position": (-rotation.T @ translation).tolist()})
        if next(lines, None) is None:
            raise ValueError("Missing COLMAP observation line")
    if not cameras or len({c["id"] for c in cameras}) != len(cameras) or len({c["img_name"] for c in cameras}) != len(cameras):
        raise ValueError("Empty or duplicate COLMAP images")
    return sorted(cameras, key=lambda c: c["img_name"])


def camera_normals(quats, viewmat):
    if not bool(torch.isfinite(quats).all()) or bool((quats.norm(dim=-1) == 0).any()):
        raise ValueError("Invalid Gaussian quaternion")
    w, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    world = torch.stack((2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)), dim=-1)
    return world[None] @ viewmat[:, :3, :3].transpose(-1, -2)


def attributes(raw, viewmat, meta, degree, wrapper, channels):
    rgb_depth = pr9.raster_colors(raw, viewmat, meta, degree, wrapper)
    if channels == 3:
        return rgb_depth[..., :3]
    if channels == 4:
        return rgb_depth
    return torch.cat((rgb_depth, camera_normals(raw["quats"], viewmat)), dim=-1)


def image_metrics(actual, expected):
    """Gate every logical channel, but report PSNR on RGB only."""
    result = pr9.compare_image(actual, expected)
    rgb = pr9.compare_image(actual[..., :3], expected[..., :3])
    result.update(mse=rgb["mse"], psnr=rgb["psnr"], psnr_channels="RGB", rgb_max_abs=rgb["max_abs"])
    if actual.shape[-1] >= 4:
        result["depth_max_abs"] = pr9.compare_image(actual[..., 3:4], expected[..., 3:4])["max_abs"]
    if actual.shape[-1] == 7:
        result["normal_max_abs"] = pr9.compare_image(actual[..., 4:7], expected[..., 4:7])["max_abs"]
    return result


def snapshot(run, inputs, upstream, expected=None, keep_reference=True):
    reference, metrics, summary = pr9._snapshot(run, inputs, upstream, expected, keep_reference)
    image = reference["tensors"]["image"]
    metrics["image"] = image_metrics(image, expected["tensors"]["image"] if expected else image)
    return reference, metrics, summary


def validate_reference(reference, signature):
    if reference.get("signature") != signature:
        raise ValueError("Reference input contents or settings differ")
    if not all(m["alignment_passed"] for m in reference.get("repeat_noise", {}).values()) or not reference.get("repeat_noise"):
        raise ValueError("Reference repeat correctness failed or missing")


def upstream_values(seed, camera_index, width, height, channels, device):
    case_seed = (seed + camera_index * 1_000_003 + width * 1009 + height) % (2**63 - 1)
    generator = torch.Generator(device=device).manual_seed(case_seed)
    return case_seed, tuple(torch.randn((1, height, width, c), generator=generator, device=device) for c in (channels, 1))


def run_case(args, raw, camera, index, size, degree, settings, identity, rasterization, wrapper):
    width, height = size
    view_np, K_np = pr9.camera_matrices(camera, width, height)
    view, K = torch.from_numpy(view_np)[None].cuda(), torch.from_numpy(K_np)[None].cuda()
    # The full renderer is only a preparation helper, never part of A/B/C timing.
    with torch.no_grad():
        setup_image, setup_alpha, meta = rasterization(
            raw["means"], torch.nn.functional.normalize(raw["quats"], dim=-1),
            raw["log_scales"].exp(), raw["opacity_logits"].sigmoid(), raw["sh_coefficients"],
            view, K, width, height, sh_degree=degree, render_mode="RGB+D", **settings)
        del setup_image, setup_alpha
        colors = attributes(raw, view, meta, degree, wrapper, args.channels)
        inputs = {name: value.detach().contiguous().requires_grad_(True) for name, value in
                  {"means2d": meta["means2d"], "conics": meta["conics"], "opacities": meta["opacities"], "colors": colors}.items()}
        raster_meta = {key: meta[key] for key in ("flatten_ids", "isect_offsets")}
        del meta, colors

    def run(_train):
        image, alpha = wrapper.rasterize_to_pixels(
            inputs["means2d"], inputs["conics"], inputs["colors"], inputs["opacities"],
            width, height, args.tile_size, raster_meta["isect_offsets"], raster_meta["flatten_ids"],
            packed=False, absgrad=True)
        return image, alpha, {**raster_meta, "means2d": inputs["means2d"]}

    seed, upstream = upstream_values(args.seed, index, width, height, args.channels, "cuda")
    signature = {**identity, "protocol": PROTOCOL, "settings": settings, "scope": "raster",
                 "channels": args.channels, "normal_convention": NORMAL_CONVENTION,
                 "compositing": "alpha_weighted_sum_no_background", "degree": degree,
                 "camera_index": index, "camera": camera, "resolution": list(size),
                 "scaled_K": K_np.tolist(), "viewmat": view_np.tolist(), "seed": seed,
                 "upstream": f"seeded_cuda_randn_image{args.channels}_alpha1_v1"}
    name = f"cam{index}-{width}x{height}-c{args.channels}-seed{seed}.pt"
    expected = None
    if args.compare_reference_dir:
        expected = torch.load(args.compare_reference_dir / name, map_location="cpu", weights_only=True)
        validate_reference(expected, signature)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    forward = pr9._measure(run, inputs, upstream, args.warmup, args.repeats, False, torch.cuda, events)
    training = pr9._measure(run, inputs, upstream, args.warmup, args.repeats, True, torch.cuda, events)
    first, parity, summary = snapshot(run, inputs, upstream, expected)
    second, noise, _ = snapshot(run, inputs, upstream, first, keep_reference=False)
    del second
    finite = {"forward_only": forward["finite_checks"], "training": training["finite_checks"]}
    passed = all(m["alignment_passed"] for m in [*parity.values(), *noise.values()]) and all(all(c.values()) for c in finite.values())
    if args.save_reference_dir and passed:
        args.save_reference_dir.mkdir(parents=True, exist_ok=True)
        torch.save({**first, "signature": signature, "repeat_noise": noise, "provenance": args.provenance}, args.save_reference_dir / name)
    return {"status": "ok" if passed else "correctness_failed", "signature": signature,
            "reference_file": name, "alignment_passed": passed if expected else None,
            "correctness_passed": passed, "parity": parity if expected else None, "repeat_noise": noise,
            "baseline_repeat_noise": expected["repeat_noise"] if expected else None,
            "baseline_provenance": expected["provenance"] if expected else None,
            "finite_checks": finite, **summary,
            "timing": {"forward_only": forward["forward"], "training_forward": training["forward"],
                       "backward": training["backward"], "combined": training["combined"]}}


def main():
    args = parse_args()
    cameras = colmap_cameras(args.colmap)
    if max(args.camera_indices) >= len(cameras):
        raise ValueError("Camera index outside COLMAP image list")
    # Content-addressed adapter JSON is always outside the source/reference tree.
    import hashlib
    camera_text = json.dumps(cameras, sort_keys=True, allow_nan=False) + "\n"
    camera_hash = hashlib.sha256(camera_text.encode()).hexdigest()
    camera_path = Path("/tmp/opencode") / f"warp-private-cameras-{camera_hash}.json"
    camera_path.write_text(camera_text)
    identity = {"ply_sha256": pr9._sha(args.ply), "cameras_sha256": camera_hash,
                "colmap_sha256": {name: pr9._sha(args.colmap / name) for name in ("cameras.txt", "images.txt")},
                "script_sha256": pr9._sha(Path(__file__)), "pr9_sha256": pr9._sha(BLOG / "benchmark_megapixels.py"),
                "scene_loader_sha256": pr9._sha(BLOG / "gsplat_blog_benchmark.py"),
                "torch": str(torch.__version__), "cuda": torch.version.cuda}
    args.provenance = {"extension_path": str(args.extension.resolve()), "extension_sha256": pr9._sha(args.extension),
                       "label": args.label, "revision": args.revision}
    pr9._load_extension(args.extension)
    from gsplat.cuda import _wrapper
    from gsplat.rendering import rasterization
    sys.path.insert(0, str(BLOG))
    from gsplat_blog_benchmark import inspect_scene_metadata, load_scene

    torch.set_num_threads(8)
    metadata = inspect_scene_metadata(args.ply, camera_path, 0, args.camera_indices[0])
    (means, quats, scales), opacity, sh, view, K, _, _, degree, total = load_scene(
        args.ply, camera_path, 0, args.camera_indices[0], scene_metadata=metadata)
    del metadata, view, K
    raw = dict(means=means, quats=quats, log_scales=scales, opacity_logits=opacity, sh_coefficients=sh)
    identity.update(points_total=total, points_used=len(means),
                    wrapper_sha256=pr9._sha(Path(_wrapper.__file__)),
                    rendering_sha256=pr9._sha(ROOT / "gsplat/rendering.py"))
    settings = dict(packed=False, absgrad=True, tile_size=args.tile_size, compact_box=True, compact_box_impl="sweep")
    report = {"protocol": PROTOCOL, "provenance": args.provenance, "identity": identity,
              "camera_json": str(camera_path), "gpu": torch.cuda.get_device_name(),
              "config": {"warmup": args.warmup, "repeats": args.repeats, "channels": args.channels,
                         "physical_channels": 8 if args.channels == 7 else args.channels,
                         "max_points": 0, "forward_max_abs": pr9.FORWARD_MAX_ABS, "gradient_rel_l2": pr9.GRADIENT_REL_L2},
              "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in args.camera_indices:
        for resolution in args.resolutions:
            size = ((cameras[index]["width"], cameras[index]["height"]) if resolution == "native"
                    else (3840, 2160) if resolution == "4k" else resolution)
            record = run_case(args, raw, cameras[index], index, size, degree, settings, identity, rasterization, _wrapper)
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(r["correctness_passed"] for r in report["records"]):
        raise SystemExit("Correctness gates failed; see output JSON")


if __name__ == "__main__":
    main()
