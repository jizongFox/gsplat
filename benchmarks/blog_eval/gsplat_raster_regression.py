"""Differential screen-space rasterizer checks, with one extension per process.

Example (use the same Python/PYTHONPATH for both commands)::

    python benchmarks/blog_eval/gsplat_raster_regression.py \
        --extension /tmp/opencode/gsplat-blog-variants/baseline.so --save baseline.pt
    python benchmarks/blog_eval/gsplat_raster_regression.py \
        --extension /path/to/candidate.so --compare baseline.pt --output report.json

The parent never imports gsplat. Each invocation starts a fresh worker because
loading two extension binaries in one process duplicates pybind enum registrations.
Tile masks are intentionally excluded: the inherited baseline leaves masked-tile
alpha uninitialized. This suite neither fixes nor treats that bug as a contract.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import typing as t
from pathlib import Path


def cases() -> t.List[t.Dict[str, t.Any]]:
    result = []
    for channels in (1, 3, 4, 8):
        for absgrad in (False, True):
            for tile_size in (8, 16):
                for scenario in (
                    "ordinary",
                    "background",
                    "multicamera",
                    "cutoff",
                    "clamp",
                    "empty_intersections",
                    "occlusion",
                ):
                    result.append(
                        dict(
                            name=f"c{channels}_abs{int(absgrad)}_t{tile_size}_{scenario}",
                            channels=channels,
                            absgrad=absgrad,
                            tile_size=tile_size,
                            scenario=scenario,
                            cameras=3 if scenario == "multicamera" else 1,
                            background=scenario != "ordinary",
                            width=29,
                            height=23,
                            count=192 if scenario == "occlusion" else 24,
                        )
                    )
    # Focused additions, not another Cartesian product. Keep existing cases
    # in their original order because their seeds depend on the case index.
    for scenario in (
        "zero_n",
        "batch_boundary",
        "packed_chunks",
        "sparse_tiles",
        "finite_overflow",
        "rejected_nonfinite_upstream",
        "clamp_boundary",
    ):
        for absgrad in (False, True):
            result.append(
                dict(
                    name=f"target_{scenario}_abs{int(absgrad)}",
                    channels=4,
                    absgrad=absgrad,
                    tile_size=16,
                    scenario=scenario,
                    cameras=3 if scenario == "packed_chunks" else 1,
                    background=scenario != "rejected_nonfinite_upstream",
                    width=49 if scenario == "sparse_tiles" else 29,
                    height=37 if scenario == "sparse_tiles" else 23,
                    count=(
                        0
                        if scenario == "zero_n"
                        else 600 if scenario == "batch_boundary" else 4
                    ),
                )
            )
    # A tile4 block has only 16 lanes; tile6 has a partial final warp (36
    # threads). Exercise nonempty forward/backward work, backgrounds, early
    # termination, and the 8-channel fallback without multiplying every case.
    partial_warps = [
        (4, channels, absgrad) for channels in (1, 3, 4, 8) for absgrad in (False, True)
    ] + [(6, 3, False), (6, 4, True)]
    for tile_size, channels, absgrad in partial_warps:
        result.append(
            dict(
                name=f"target_partial_warp_t{tile_size}_c{channels}_abs{int(absgrad)}",
                channels=channels,
                absgrad=absgrad,
                tile_size=tile_size,
                scenario="occlusion",
                cameras=1,
                background=True,
                width=29,
                height=23,
                count=192,
            )
        )
    # Append channel-2 equivalents to exercise the ten-component layout while
    # preserving every original case's name, configuration and index-derived seed.
    result.extend(
        dict(config, channels=2, name=config["name"].replace("c1_", "c2_"))
        for config in tuple(result)
        if config["channels"] == 1
    )
    # Append without changing any PR9 case/index-derived seed. Exercise the
    # narrow forward shared-memory layout at the largest supported block,
    # including both sides of its 1024-splat staging boundary.
    for channels in (1, 2, 3, 4):
        for absgrad in (False, True):
            result.append(
                dict(
                    name=f"target_staging_t32_c{channels}_abs{int(absgrad)}",
                    channels=channels,
                    absgrad=absgrad,
                    tile_size=32,
                    scenario="staging_boundary",
                    cameras=1,
                    background=True,
                    width=49,
                    height=37,
                    count=1300,
                )
            )
    return result


def provenance(extension: Path, wrapper: Path) -> t.Dict[str, t.Any]:
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    def git(*arguments):
        process = subprocess.run(
            ["git", "-C", str(wrapper.parent), *arguments],
            capture_output=True,
            check=False,
        )
        return process.stdout if process.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    diff = git("diff", "--binary", "HEAD", "--", ".")
    return dict(
        extension_path=str(extension.resolve()),
        extension_sha256=digest(extension),
        wrapper_path=str(wrapper.resolve()),
        wrapper_sha256=digest(wrapper),
        source_revision=revision.decode().strip() if revision is not None else None,
        source_diff_sha256=(
            hashlib.sha256(diff).hexdigest() if diff is not None else None
        ),
        source_diff_scope="tracked changes under wrapper directory (gsplat/cuda)",
        source_note="Source snapshot at run time; not proof of binary build inputs.",
    )


def run_case(
    config: t.Dict[str, t.Any], seed: int, wrapper: t.Any, stream: t.Any = None
) -> t.Dict:
    import torch

    # Scope every CUDA operation, including transfers and binning, to the
    # selected stream. No SH or projection code participates in these fixtures.
    with torch.cuda.stream(stream):
        tensors = raster_case(config, seed, wrapper)
    if stream is not None:
        stream.synchronize()
    return {name: value.detach().cpu() for name, value in tensors.items()}


def raster_case(config: t.Dict[str, t.Any], seed: int, wrapper: t.Any) -> t.Dict:
    import torch

    # Generate on CPU so both extensions receive bitwise-identical inputs and
    # upstream gradients, independent of device RNG implementations.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cameras, count, channels = (
        config["cameras"],
        config["count"],
        config["channels"],
    )
    width, height, tile_size = config["width"], config["height"], config["tile_size"]

    def rand(*shape):
        return torch.rand(shape, generator=generator, dtype=torch.float32)

    means = rand(cameras, count, 2) * torch.tensor([width + 8, height + 8]) - 4
    # Strictly positive definite, including nonzero cross terms.
    diagonal = 0.04 + 0.16 * rand(cameras, count, 2)
    cross = (rand(cameras, count) - 0.5) * 0.025
    conics = torch.stack((diagonal[..., 0], cross, diagonal[..., 1]), dim=-1)
    colors = rand(cameras, count, channels) * 1.5 - 0.2
    opacities = 0.05 + 0.85 * rand(cameras, count)
    scenario = config["scenario"]
    if scenario == "cutoff":
        # At the central pixel sigma=0, straddle the 1/255 alpha cutoff.
        means[:] = torch.tensor([width // 2 + 0.5, height // 2 + 0.5])
        values = torch.tensor([0.0, 0.99 / 255, 1.0 / 255, 1.01 / 255, 0.02, 0.1])
        opacities[:] = values.repeat(math.ceil(count / len(values)))[:count]
    elif scenario == "clamp":
        means[:] = torch.tensor([width // 2 + 0.5, height // 2 + 0.5])
        values = torch.tensor([0.989, 0.99, 0.991, 1.0, 1.2, 0.2])
        opacities[:] = values.repeat(math.ceil(count / len(values)))[:count]
    elif scenario == "occlusion":
        means[:] = torch.tensor([width / 2, height / 2]) + (
            rand(cameras, count, 2) - 0.5
        )
        conics[:] = torch.tensor([0.008, 0.001, 0.009])
        opacities[:] = 0.85 + 0.14 * rand(cameras, count)
    elif scenario == "empty_intersections":
        means[:] = -1000
    elif scenario == "batch_boundary":
        means[:] = torch.tensor([0.5, 0.5])
        conics[:] = torch.tensor([1e-8, 0.0, 1e-8])
        # T crosses 1e-4 at splats 255/257 (either side of the 256-thread
        # batch), with 600 depth-sorted splats available. Small spatial decay
        # keeps every pixel close to this boundary, not just the center.
        termination = 255 if not config["absgrad"] else 257
        opacities[:] = 1.0 - math.exp(math.log(1e-4) / (termination - 0.5))
    elif scenario == "staging_boundary":
        means[:] = torch.tensor([0.5, 0.5])
        conics[:] = torch.tensor([1e-8, 0.0, 1e-8])
        # Exclusive termination: allow several actual color contributions
        # from batch two, rather than merely loading then rejecting its head.
        termination = 1023 if not config["absgrad"] else 1033
        opacities[:] = 1.0 - math.exp(math.log(1e-4) / (termination - 0.5))
    elif scenario in ("finite_overflow", "rejected_nonfinite_upstream"):
        means[:] = torch.tensor([0.5, 0.5])
        conics[:] = torch.tensor([1e38, 0.0, 1e38])
        opacities[:] = 0.2
        # Only pixel (0,0) contributes. Other lanes overflow sigma and/or
        # conic*delta, but must be rejected before evaluating gradients.
    elif scenario == "clamp_boundary":
        means[:] = torch.tensor([0.5, 0.5])
        conics[:] = torch.tensor([0.2, 0.01, 0.2])
        opacities[:] = torch.tensor([0.9989, 0.999, 0.9991, 1.1])
    elif scenario == "sparse_tiles":
        means[:] = torch.tensor([[[2.5, 2.5], [3.5, 3.5], [48.5, 36.5], [-100, -100]]])
        conics[:] = torch.tensor([2.0, 0.0, 2.0])

    packed = scenario == "packed_chunks"
    if packed:
        # Dense per-camera screen inputs flattened into packed nnz storage;
        # explicit ids retain camera separation in binning.
        means = means.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, channels)
        opacities = opacities.reshape(-1)

    inputs = {
        name: value.cuda().contiguous().requires_grad_()
        for name, value in dict(
            means2d=means, conics=conics, colors=colors, opacities=opacities
        ).items()
    }
    background = None
    if config["background"]:
        background = rand(cameras, channels).cuda().requires_grad_()
        inputs["backgrounds"] = background

    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    # Conservative bounds for these synthetic covariances. Binning is fixed,
    # not differentiated, just as in the normal rendering pipeline.
    radii = torch.full((cameras, count), 48, dtype=torch.int32, device="cuda")
    if scenario == "sparse_tiles":
        radii.fill_(4)
    depths = (
        torch.arange(1, count + 1, dtype=torch.float32, device="cuda")
        .expand(cameras, -1)
        .contiguous()
    )
    with torch.no_grad():
        bin_options = {}
        if packed:
            radii = radii.flatten()
            depths = depths.flatten()
            bin_options = dict(
                packed=True,
                n_cameras=cameras,
                camera_ids=torch.arange(
                    cameras, device="cuda", dtype=torch.int64
                ).repeat_interleave(count),
                gaussian_ids=torch.arange(
                    count, device="cuda", dtype=torch.int64
                ).repeat(cameras),
            )
        _, isect_ids, flatten_ids = wrapper.isect_tiles(
            inputs["means2d"],
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            **bin_options,
        )
        offsets = wrapper.isect_offset_encode(
            isect_ids, cameras, tile_width, tile_height
        )
    if scenario in ("empty_intersections", "zero_n") and flatten_ids.numel() != 0:
        raise AssertionError("Empty-intersection fixture unexpectedly intersects tiles")
    snapshots = (offsets.clone(), flatten_ids.clone()) if packed else None
    chunks = ((0, 3), (3, 4)) if packed else ((0, channels),)
    tensors = {}
    for chunk, (start, stop) in enumerate(chunks):
        for value in inputs.values():
            value.grad = None
        image, alpha = wrapper.rasterize_to_pixels(
            inputs["means2d"],
            inputs["conics"],
            inputs["colors"][..., start:stop].contiguous(),
            inputs["opacities"],
            width,
            height,
            tile_size,
            offsets,
            flatten_ids,
            backgrounds=(
                background[:, start:stop].contiguous()
                if background is not None
                else None
            ),
            packed=packed,
            absgrad=config["absgrad"],
        )
        upstream = tuple(
            torch.randn(value.shape, generator=generator, dtype=torch.float32).cuda()
            for value in (image, alpha)
        )
        if scenario == "rejected_nonfinite_upstream":
            # No background gradient: its full-image reduction would correctly
            # propagate NaNs. Only rejected pixels get poisoned; the accepted
            # lane has fixed finite upstreams. All saved outputs/GS gradients
            # must remain finite, including absgrad.
            for value in upstream:
                value.fill_(float("nan"))
                value[:, 0, 0, :] = 1.0
        torch.autograd.backward((image, alpha), upstream)
        if scenario == "staging_boundary":
            used_second_batch = bool((inputs["colors"].grad[:, 1024:] != 0).any())
            if used_second_batch != config["absgrad"]:
                raise AssertionError("Staging fixture did not reach its intended batch")
        prefix = f"chunk{chunk}/" if packed else ""
        tensors.update({prefix + "image": image, prefix + "alpha": alpha})
        for name, value in inputs.items():
            if value.grad is None:
                raise AssertionError(f"Missing gradient: {name}")
            tensors[f"grad/{prefix}{name}"] = value.grad
        if config["absgrad"]:
            absolute = getattr(inputs["means2d"], "absgrad", None)
            if absolute is None:
                raise AssertionError("Missing means2d.absgrad")
            tensors[f"grad/{prefix}means2d.absgrad"] = absolute
        if snapshots is not None:
            if not all(
                torch.equal(a, b) for a, b in zip(snapshots, (offsets, flatten_ids))
            ):
                raise AssertionError("Rasterization mutated reused intersection lists")
    return tensors


def compare_tensors(actual: t.Dict, expected: t.Dict) -> t.Dict:
    import torch

    if actual.keys() != expected.keys():
        raise ValueError("Reference tensor names differ")
    results = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Reference shape/dtype differs: {name}")
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        delta = value.double() - reference.double()
        max_abs = float(delta.abs().max()) if finite and delta.numel() else 0.0
        rel_l2 = (
            float(delta.norm() / reference.double().norm().clamp_min(1e-30))
            if finite
            else None
        )
        gradient = name.startswith("grad/")
        results[name] = dict(
            finite=finite,
            max_abs=max_abs if finite else None,
            rel_l2=rel_l2,
            passed=finite and (rel_l2 <= 1e-4 if gradient else max_abs <= 2e-5),
        )
    return results


def worker(args: argparse.Namespace) -> int:
    import gsplat
    import torch

    # Match gsplat_blog_benchmark.py: inject the selected module BEFORE the
    # backend import, which otherwise loads the installed/JIT extension.
    spec = importlib.util.spec_from_file_location(
        "gsplat.csrc", args.extension.resolve()
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load extension: {args.extension}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat.csrc"] = extension
    gsplat.csrc = extension
    from gsplat.cuda import _backend, _wrapper

    if _backend._C is not extension:
        raise RuntimeError("gsplat backend did not use the requested extension")
    metadata = provenance(args.extension, Path(_wrapper.__file__))
    metadata["stream"] = "nondefault" if args.nondefault_stream else "default"
    stream = torch.cuda.Stream() if args.nondefault_stream else None
    configs = cases()
    reference = None
    if args.compare:
        reference = torch.load(args.compare, map_location="cpu", weights_only=True)
        if (
            reference["version"] != 2
            or reference["seed"] != args.seed
            or reference["configs"] != configs
        ):
            raise ValueError("Reference suite version, seed, or configuration differs")
        if set(reference["tensors"]) != {config["name"] for config in configs}:
            raise ValueError("Reference case names differ")
    report = dict(
        extension=str(args.extension.resolve()),
        reference=str(args.compare) if args.compare else None,
        seed=args.seed,
        provenance=metadata,
        baseline_provenance=reference["provenance"] if reference else metadata,
        candidate_provenance=metadata if reference else None,
        thresholds=dict(forward_max_abs=2e-5, gradient_rel_l2=1e-4),
        cases={},
        passed=True,
    )
    saved = {}
    for index, config in enumerate(configs):
        name = config["name"]
        actual = run_case(config, args.seed + index, _wrapper, stream=stream)
        expected = reference["tensors"][name] if reference else actual
        metrics = compare_tensors(actual, expected)
        passed = all(metric["passed"] for metric in metrics.values())
        report["cases"][name] = dict(passed=passed, tensors=metrics)
        report["passed"] = report["passed"] and passed
        if args.save:
            saved[name] = actual
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    if args.save and report["passed"]:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                version=2,
                seed=args.seed,
                configs=configs,
                tensors=saved,
                provenance=metadata,
            ),
            args.save,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(configs)} cases: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--save", type=Path)
    action.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--nondefault-stream",
        action="store_true",
        help="Run raster-only fixtures on a nondefault CUDA stream; reuse either stream's reference",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.extension.is_file():
        parser.error(f"Extension does not exist: {args.extension}")
    if args.compare and not args.compare.is_file():
        parser.error(f"Reference does not exist: {args.compare}")
    if args.worker:
        return worker(args)
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
