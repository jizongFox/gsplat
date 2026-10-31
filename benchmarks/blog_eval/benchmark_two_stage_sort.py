"""Two-stage-sort evaluation using PR9 timing/parity helpers without changing them.

One binary per fresh process. Large exact-intersection/gradient references stay
outside Git. The full-render workload is the unchanged PR9 RGB+ED/absgrad setup.
"""

import argparse
import functools
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch

import benchmark_megapixels as protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--camera-indices", nargs="+", type=int, default=[0, 623, 1247])
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=protocol.parse_resolution,
        default=[(1280, 720), (3840, 2160)],
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--binning-only", action="store_true")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--save-reference-dir", type=Path)
    actions.add_argument("--compare-reference-dir", type=Path)
    actions.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.profile_repeats < 0:
        parser.error("Invalid repeat counts")
    for directory in (args.save_reference_dir, args.compare_reference_dir):
        if directory and directory.resolve().is_relative_to(
            Path(__file__).resolve().parents[2]
        ):
            parser.error("Store tensor references outside the repository")
    return args


def binning_case(args, meta, width, height, camera, wrapper, identity):
    tile_width, tile_height = math.ceil(width / 16), math.ceil(height / 16)
    data = {
        name: meta[name]
        for name in ("means2d", "radii", "depths", "conics", "opacities")
    }
    signature = dict(
        identity,
        camera=camera,
        resolution=[width, height],
        tile_size=16,
        compact_box=True,
        compact_box_impl="sweep",
    )

    def run():
        return wrapper.isect_tiles(
            data["means2d"],
            data["radii"],
            data["depths"],
            16,
            tile_width,
            tile_height,
            conics=data["conics"],
            opacities=data["opacities"],
            compact_box=True,
            compact_box_impl="sweep",
            two_stage_sort=args.two_stage,
        )

    for _ in range(args.warmup):
        result = run()
        del result
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    gpu_samples, wall_samples = [], []
    for _ in range(args.repeats):
        wall_start = time.perf_counter()
        start.record()
        result = run()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
        gpu_samples.append(start.elapsed_time(end))
        del result
    result = run()
    counts, keys, ids = result
    output = dict(
        api_cuda_event_ms=protocol._stats(gpu_samples),
        synchronized_wall_ms=protocol._stats(wall_samples),
        input_gaussians=counts.numel(),
        visible_gaussians=int((counts > 0).sum()),
        intersections=keys.numel(),
        public_dtypes=[str(t.dtype) for t in result],
        baseline_sort_bits=32 + (tile_width * tile_height).bit_length() + 1,
    )
    if not args.skip_validation:
        name = f"binning-cam{camera}-{width}x{height}.pt"
        if args.compare_reference_dir:
            reference = torch.load(args.compare_reference_dir / name, weights_only=True)
            if reference["signature"] != signature:
                raise ValueError("Binning reference identity differs")
            output["exact_output_parity"] = all(
                torch.equal(actual.cpu(), expected)
                for actual, expected in zip(result, reference["outputs"])
            )
            if not output["exact_output_parity"]:
                raise RuntimeError(
                    "Two-stage output differs from baseline (keys/IDs/counts)"
                )
        else:
            args.save_reference_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"signature": signature, "outputs": [t.cpu() for t in result]},
                args.save_reference_dir / name,
            )
    del counts, keys, ids, result
    if args.profile_repeats:
        # Diagnostic only: CUDA radix kernels, not the whole binning API. This
        # intentionally excludes visibility compaction, scans and reconstruction.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(args.profile_repeats):
                result = run()
                del result
            torch.cuda.synchronize()
        events = [
            dict(name=e.key, calls=e.count, total_device_ms=e.device_time_total / 1000)
            for e in prof.key_averages()
            if "radix" in e.key.lower() and e.device_time_total > 0
        ]
        output["radix_profile"] = dict(
            repeats=args.profile_repeats,
            kernels=events,
            summed_device_ms_per_call=sum(e["total_device_ms"] for e in events)
            / args.profile_repeats,
        )
    return output


def main():
    args = parse_args()
    torch.set_num_threads(8)
    extension = protocol._load_extension(args.extension)
    from gsplat.cuda import _wrapper
    from gsplat.rendering import rasterization
    from gsplat_blog_benchmark import inspect_scene_metadata, load_scene

    metadata = inspect_scene_metadata(args.ply, args.cameras, 0, args.camera_indices[0])
    (means, quats, scales), opacity, sh, _, _, _, _, degree, total = load_scene(
        args.ply, args.cameras, 0, args.camera_indices[0], scene_metadata=metadata
    )
    del metadata
    raw = dict(
        means=means,
        quats=quats,
        log_scales=scales,
        opacity_logits=opacity,
        sh_coefficients=sh,
    )
    for tensor in raw.values():
        tensor.requires_grad_(True)
    cameras = json.loads(args.cameras.read_text())
    settings = dict(
        packed=False,
        render_mode="RGB+ED",
        absgrad=True,
        tile_size=16,
        compact_box=True,
        compact_box_impl="sweep",
    )
    identity = dict(
        ply_sha256=protocol._sha(args.ply),
        cameras_sha256=protocol._sha(args.cameras),
        points_total=total,
        points_used=means.shape[0],
    )
    # Strategy is provenance, not reference-workload identity: the output
    # contract is required to be identical across strategies.
    renderer = functools.partial(rasterization, two_stage_sort=args.two_stage)
    args.extension_sha256 = protocol._sha(args.extension)
    args.revision = "two-stage" if args.two_stage else "a7050c6-baseline"
    args.label = args.revision
    args.scope = "full"
    report = dict(
        two_stage=args.two_stage,
        extension_sha256=args.extension_sha256,
        script_sha256=protocol._sha(Path(__file__)),
        inputs=identity,
        config=settings,
        environment=dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            extension=str(extension.__file__),
        ),
        warmup=args.warmup,
        repeats=args.repeats,
        records=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for camera in args.camera_indices:
        for width, height in args.resolutions:
            view, intrinsic = protocol.camera_matrices(cameras[camera], width, height)
            with torch.no_grad():
                image, alpha, meta = renderer(
                    means,
                    torch.nn.functional.normalize(quats, dim=-1),
                    scales.exp(),
                    opacity.sigmoid(),
                    sh,
                    torch.from_numpy(view)[None].cuda(),
                    torch.from_numpy(intrinsic)[None].cuda(),
                    width,
                    height,
                    sh_degree=degree,
                    **settings,
                )
                del image, alpha
                binned = binning_case(
                    args, meta, width, height, camera, _wrapper, identity
                )
                del meta
            record = dict(
                camera_index=camera, resolution=[width, height], binning=binned
            )
            if not args.binning_only:
                record["full_renderer"] = protocol._case(
                    args,
                    raw,
                    cameras[camera],
                    camera,
                    width,
                    height,
                    "full",
                    degree,
                    settings,
                    identity,
                    renderer,
                    _wrapper,
                    {},
                )
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                f"view{camera} {width}x{height}: binning{binned['api_cuda_event_ms']['mean_ms']:.3f}ms",
                flush=True,
            )
            protocol._clear(raw)
            gc.collect()
            torch.cuda.empty_cache()
    failures = [
        r
        for r in report["records"]
        if r.get("full_renderer", {}).get("alignment_passed") is False
    ]
    if failures:
        raise SystemExit(
            "Full-renderer parity gate failed; reports retained, thresholds unchanged"
        )


if __name__ == "__main__":
    main()
