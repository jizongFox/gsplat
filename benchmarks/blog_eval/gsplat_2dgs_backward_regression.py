"""Compare 2DGS raster binaries in separate processes, including every VJP.

From the repository root, use the same interpreter for both invocations::

    python -m benchmarks.blog_eval.gsplat_2dgs_backward_regression \
        --extension /path/to/baseline.so --save /tmp/2dgs-baseline.pt
    python -m benchmarks.blog_eval.gsplat_2dgs_backward_regression \
        --extension /path/to/candidate.so --compare /tmp/2dgs-baseline.pt \
        --output /tmp/2dgs-candidate.json --nondefault-stream

Optional --warmup 30 --repeats 80 measures synthetic forward/backward API
latency with CUDA events, including backward output allocation/zeroing. These
are detached raster fixtures, not full-scene or training benchmarks. Compare
the base, tail-only commit, and combined commit to isolate the two changes.

The inherited densification proxy reads concurrently accumulated ray gradients.
It is deliberately included in the strict gradient gate; a mismatch or baseline
repeat instability must be investigated, not removed from the passing criteria.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

from benchmarks.blog_eval.gsplat_raster_regression import compare_tensors, provenance


PROTOCOL = "2dgs_backward_v1"


def cases():
    result = []

    def add(
        name,
        channels=4,
        tile_size=16,
        absgrad=True,
        scenario="ordinary",
        cameras=1,
        count=37,
        packed=False,
        background=True,
        distortion=True,
        median=True,
    ):
        result.append(
            dict(
                name=name,
                channels=channels,
                tile_size=tile_size,
                absgrad=absgrad,
                scenario=scenario,
                cameras=cameras,
                count=count,
                packed=packed,
                background=background,
                distortion=distortion,
                median=median,
                width=29,
                height=23,
            )
        )

    for channels in (1, 2, 3, 4, 8):
        for absgrad in (False, True):
            for tile_size in (8, 16):
                add(
                    f"ordinary_c{channels}_abs{int(absgrad)}_t{tile_size}",
                    channels=channels,
                    absgrad=absgrad,
                    tile_size=tile_size,
                )
    for scenario, count in (
        ("occlusion", 600),
        ("boundary_255", 600),
        ("boundary_257", 600),
        ("boundary_259", 600),
        ("no_contributors", 600),
        ("empty_tiles", 37),
        ("empty_intersections", 37),
        ("zero_n", 0),
        ("cutoff", 37),
        ("clamp", 37),
    ):
        for packed in (False, True):
            add(
                f"{scenario}_packed{int(packed)}",
                scenario=scenario,
                count=count,
                cameras=2,
                packed=packed,
            )
    for tile_size in (4, 6):
        for channels in (3, 4, 8):
            add(
                f"partial_t{tile_size}_c{channels}",
                tile_size=tile_size,
                channels=channels,
                scenario="occlusion",
                count=192,
            )
    add("no_background", background=False)
    add("no_auxiliary_losses", distortion=False, median=False)
    for scenario in ("alpha_only", "normals_only", "distortion_only", "median_only"):
        add(scenario, scenario=scenario)
    return result


def select_cases(names):
    configs = cases()
    if not names:
        return configs
    unknown = set(names) - {config["name"] for config in configs}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [config for config in configs if config["name"] in names]


def reference_identity(configs, seed):
    return dict(
        protocol=PROTOCOL,
        configs=configs,
        seed=seed,
        fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def validate_reference(reference, identity):
    if reference.get("identity") != identity:
        raise ValueError("Reference protocol, fixture source, seed or cases differ")
    if set(reference.get("tensors", {})) != {
        config["name"] for config in identity["configs"]
    }:
        raise ValueError("Reference case names differ")


def report_passed(records):
    return bool(records) and all(
        bool(record["tensors"])
        and all(metric["passed"] for metric in record["tensors"].values())
        for record in records.values()
    )


def make_inputs(config, seed):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    cameras, count, channels = (config[key] for key in ("cameras", "count", "channels"))
    width, height, tile_size = (config[key] for key in ("width", "height", "tile_size"))

    def rand(*shape):
        return torch.rand(shape, generator=generator, dtype=torch.float32)

    means = rand(cameras, count, 2) * torch.tensor([width, height])
    depth = torch.arange(count, dtype=torch.float32).expand(cameras, -1) * 0.001 + 2
    scale = 4 + rand(cameras, count) * 8
    scale[:, ::2] = 0.3  # Exercise the projected 2D filter and its absgrad.
    opacity = 0.05 + rand(cameras, count) * 0.7
    scenario = config["scenario"]
    if scenario == "occlusion" or scenario.startswith("boundary_"):
        means[:] = torch.tensor([width / 2, height / 2])
        scale[:] = 1e4
        if scenario == "occlusion":
            opacity[:] = 0.9
        else:
            boundary = int(scenario.rsplit("_", 1)[1])
            opacity[:] = 1 - math.exp(math.log(1e-4) / (boundary - 0.5))
    elif scenario == "no_contributors":
        opacity[:] = 0.5 / 255
    elif scenario in ("cutoff", "clamp"):
        means[:] = torch.tensor([0.5, 0.5])
        values = (
            [0, 0.99 / 255, 1 / 255, 1.01 / 255, 0.05]
            if scenario == "cutoff"
            else [0.9989, 0.999, 0.9991, 1.1, 0.2]
        )
        opacity[:] = torch.tensor(values).repeat(math.ceil(count / len(values)))[:count]

    rays = torch.zeros(cameras, count, 3, 3)
    rays[..., 0, 0] = scale * depth
    rays[..., 1, 1] = scale * depth * 1.1
    rays[..., 0, 1] = scale * depth * 0.05
    rays[..., :2, 2] = means * depth[..., None]
    rays[..., 2, 2] = depth
    colors = rand(cameras, count, channels)
    colors[..., -1] = depth  # The 2DGS distortion/median depth channel.
    normals = rand(cameras, count, 3) * 2 - 1
    normals /= normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    densify = torch.zeros_like(means)
    tensors = [means, rays, colors, opacity, normals, densify]
    if config["packed"]:
        tensors = [value.flatten(0, 1) for value in tensors]
    tensors = [value.cuda().contiguous() for value in tensors]
    background = rand(cameras, channels).cuda() if config["background"] else None

    # Fixed conservative lists isolate the raster kernels from projection and
    # binning. Each camera/tile has ordered absolute intersection positions.
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    offsets, flatten = [], []
    for camera in range(cameras):
        for tile in range(tile_width * tile_height):
            offsets.append(len(flatten))
            empty = scenario == "empty_intersections" or (
                scenario == "empty_tiles" and tile % 2 == 0
            )
            if not empty:
                flatten.extend(camera * count + index for index in range(count))
    offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda").reshape(
        cameras, tile_height, tile_width
    )
    flatten = torch.tensor(flatten, dtype=torch.int32, device="cuda")
    upstream = [
        rand(cameras, height, width, size).cuda() - 0.5
        for size in (channels, 1, 3, 1, 1)
    ]
    if not config["distortion"]:
        upstream[3].zero_()
    if not config["median"]:
        upstream[4].zero_()
    isolated_loss = {
        "alpha_only": 1,
        "normals_only": 2,
        "distortion_only": 3,
        "median_only": 4,
    }
    if scenario in isolated_loss:
        for index, value in enumerate(upstream):
            if index != isolated_loss[scenario]:
                value.zero_()
    return tensors, background, offsets, flatten, upstream


def run_case(extension, config, seed, args, stream):
    import torch

    with torch.cuda.stream(stream):
        tensors, background, offsets, flatten, upstream = make_inputs(config, seed)
        means, rays, colors, opacity, normals, densify = tensors
        width, height, tile_size = (
            config[key] for key in ("width", "height", "tile_size")
        )
        original_offsets, original_flatten = offsets.clone(), flatten.clone()

        def forward():
            return extension.rasterize_to_pixels_fwd_2dgs(
                means,
                rays,
                colors,
                opacity,
                normals,
                background,
                None,
                width,
                height,
                tile_size,
                offsets,
                flatten,
            )

        outputs = forward()
        if len(outputs) != 7:
            raise AssertionError("Expected all seven forward outputs")

        def backward():
            return extension.rasterize_to_pixels_bwd_2dgs(
                means,
                rays,
                colors,
                opacity,
                normals,
                densify,
                background,
                None,
                width,
                height,
                tile_size,
                offsets,
                flatten,
                outputs[0],
                outputs[1],
                outputs[5],
                outputs[6],
                *upstream,
                config["absgrad"],
            )

        gradients = backward()
        if len(gradients) != 7:
            raise AssertionError("Expected all seven backward outputs")
        names = (
            "image", "alpha", "normals", "distortion", "median", "last_ids", "median_ids"
        )
        saved = dict(zip(names, outputs))
        gradient_names = (
            "means2d.absgrad", "means2d", "ray_transforms", "colors",
            "opacities", "normals", "densify"
        )
        saved.update(
            {
                f"grad/{name}": value
                for name, value in zip(gradient_names, gradients)
                if name != "means2d.absgrad" or config["absgrad"]
            }
        )
        if not (
            torch.equal(offsets, original_offsets)
            and torch.equal(flatten, original_flatten)
        ):
            raise AssertionError("Backward mutated intersection lists")

        timings = {}
        if args.repeats:
            for name, operation in (
                ("forward_api_ms", forward), ("backward_api_ms", backward)
            ):
                for _ in range(args.warmup):
                    operation()
                samples = []
                for _ in range(args.repeats):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record(stream)
                    operation()
                    end.record(stream)
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                timings[name] = dict(median=statistics.median(samples), samples=samples)
    stream.synchronize()
    return {name: value.detach().cpu() for name, value in saved.items()}, timings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--save", type=Path)
    action.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", dest="names")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nondefault-stream", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 0:
        parser.error("warmup and repeats must be nonnegative")
    protected = {
        path.resolve() for path in (args.extension, args.save, args.compare) if path
    }
    if args.output and args.output.resolve() in protected:
        parser.error("JSON output must not overwrite an extension or tensor reference")
    if args.save and args.save.resolve() == args.extension.resolve():
        parser.error("Tensor reference must not overwrite the extension")
    try:
        select_cases(args.names)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.extension.is_file():
        raise FileNotFoundError(args.extension)
    import torch

    configs = select_cases(args.names)
    identity = reference_identity(configs, args.seed)
    reference = (
        torch.load(args.compare, map_location="cpu", weights_only=True)
        if args.compare else None
    )
    if reference is not None:
        validate_reference(reference, identity)
    device = dict(
        name=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
    )
    if reference is not None and reference["device"] != device:
        raise ValueError("Reference CUDA device or runtime differs")

    import gsplat

    spec = importlib.util.spec_from_file_location("gsplat.csrc", args.extension.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load extension: {args.extension}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat.csrc"] = extension
    gsplat.csrc = extension
    stream = (
        torch.cuda.Stream() if args.nondefault_stream else torch.cuda.default_stream()
    )
    records, saved = {}, {}
    seeds = {config["name"]: args.seed + index for index, config in enumerate(cases())}
    for config in configs:
        name = config["name"]
        actual, timings = run_case(extension, config, seeds[name], args, stream)
        expected = reference["tensors"][name] if reference else actual
        metrics = compare_tensors(actual, expected)
        for tensor_name, value in actual.items():
            if not tensor_name.startswith("grad/"):
                metrics[tensor_name]["bit_identical"] = torch.equal(
                    value, expected[tensor_name]
                )
                metrics[tensor_name]["passed"] &= metrics[tensor_name]["bit_identical"]
        records[name] = dict(tensors=metrics, timings=timings)
        if args.save:
            saved[name] = actual
        passed = report_passed({name: records[name]})
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    passed = report_passed(records)
    metadata = provenance(
        args.extension, Path(gsplat.__file__).parent / "cuda" / "_wrapper.py"
    )
    report = dict(
        identity=identity,
        device=device,
        provenance=metadata,
        stream="nondefault" if args.nondefault_stream else "default",
        baseline_provenance=reference["provenance"] if reference else metadata,
        gates=dict(forward="bit-identical", gradient_relative_l2=1e-4),
        cases=records,
        passed=passed,
    )
    if args.save and passed:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(identity=identity, tensors=saved, device=device, provenance=metadata),
            args.save,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
