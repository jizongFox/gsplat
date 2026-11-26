"""PennyGS (arXiv 2611.27525) Sec. 3.4: colour staging in the subtile forward.

Arms, in one process on one build, alternating inside each repeat:

- ``default``: the default forward kernel (``forward="pixel"``);
- ``subtile``: the subtile forward (``forward="subtile"``), which stages 1-4
  channel colours in shared memory and reads wider colours per hit.

The staging sweep in ``subtile_staging.md`` was measured at commit ``a622588``,
where this script also had ``stage_all`` and ``stage_hit`` arms that switched
the staging mode through a temporary environment variable. Check out that
commit to reproduce it.

Colours model the parent application's ``override_color``: depth (1), RGB (3),
RGB + depth (4), RGB + camera normal + depth (7, padded to 8: the training
renderer), and 16 / 32 channels with extra seeded random features. Views are
rendered at their native size (scale 1) and with intrinsics and size scaled
(scale 3: 3840x2160 for 1280x720).

Scopes:

- ``raster``: `rasterize_to_pixels` forward on fixed projected Gaussians and
  intersections (CUDA events); every arm must equal ``default`` exactly;
- ``render``: the full RGB+ED + normals renderer of the parent application
  (`benchmarks/reduce_scatter8/renderer_benchmark.py::render`), forward and
  backward.
"""

import argparse
import functools
import importlib.util
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "renderer_benchmark", HERE.parent / "reduce_scatter8" / "renderer_benchmark.py"
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

# name -> (subtile masks, staging mode)
ARMS = {"default": "pixel", "subtile": "subtile"}  # rasterize_to_pixels(forward=)
CHANNELS = (1, 3, 4, 8, 16, 32)


def foreign_gpu_pids():
    """Other processes on the GPU (the machine is shared)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return {int(p) for p in out if p.isdigit() and int(p) != os.getpid()}


class GpuGuard:
    """Polls the GPU once a second while a case runs; `seen` holds any foreign
    process that appeared, so the case can be measured again."""

    def __enter__(self):
        while foreign_gpu_pids():
            time.sleep(5)
        self.seen, self._stop = set(), threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            self.seen |= foreign_gpu_pids()
            self._stop.wait(1.0)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seen |= foreign_gpu_pids()


def guarded_measure(run, warmup, repeats, retries=20):
    for attempt in range(1, retries + 1):
        with GpuGuard() as guard:
            times, check = measure(run, warmup, repeats)
        if not guard.seen:
            return times, check, {"attempts": attempt, "contended": False}
        print(f"contended by {sorted(guard.seen)}; measuring again", flush=True)
    return times, check, {"attempts": retries, "contended": True}


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


def scaled_camera(camera, scale):
    viewmat, K = rb.camera_tensors(camera)
    K = K.clone()
    K[:2] *= scale
    width, height = int(camera["width"]) * scale, int(camera["height"]) * scale
    return viewmat.cuda(), K.cuda(), width, height


def colour_sets(rgb, normals, depth, generator):
    """Override colours per channel count (the 8-channel set has 7 real ones)."""
    n = rgb.shape[0]
    extra = torch.rand((n, 25), generator=generator).cuda()
    app = torch.cat((rgb, normals, depth[:, None]), -1)
    return {
        1: depth[:, None],
        3: rgb,
        4: torch.cat((rgb, depth[:, None]), -1),
        8: app,
        16: torch.cat((app, extra[:, :9]), -1),
        32: torch.cat((app, extra), -1),
    }


def raster_cases(inputs, viewmat, K, width, height, channels, generator):
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
        tw, th = (width + 15) // 16, (height + 15) // 16
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            16,
            tw,
            th,
            conics=conics,
            opacities=opacities,
            compact_box=True,
        )
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        sets = colour_sets(rgb, normals, depths[0], generator)

    def make(c):
        colors = sets[c][None].contiguous()
        background = torch.zeros((1, colors.shape[-1]), device="cuda")

        def run(arm):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            with torch.no_grad():
                start.record()
                image, alpha = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors,
                    opacities,
                    width,
                    height,
                    16,
                    offsets,
                    flatten_ids,
                    backgrounds=background,
                    forward=ARMS[arm],
                )
                end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), {"image": image, "alpha": alpha}, {}

        return run

    return {c: make(c) for c in channels}, int(flatten_ids.numel())


def render_case(inputs, viewmat, K, width, height, generator):
    from gsplat import rasterization

    leaves = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    upstream = None

    def run(arm):
        nonlocal upstream
        raster = functools.partial(rasterization, rasterize_forward=ARMS[arm])
        for leaf in leaves.values():
            leaf.grad = None
        torch.cuda.synchronize()
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out, alpha, _, _, _ = rb.render(raster, leaves, viewmat, K, width, height, True)
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
        return (
            (start.elapsed_time(mid), mid.elapsed_time(end)),
            {"image": out.detach(), "alpha": alpha.detach()},
            grads,
        )

    return run


def measure(run, warmup, repeats):
    times = {a: [] for a in ARMS}
    check, reference = {}, None
    for repeat in range(warmup + repeats):
        for arm in ARMS:
            ms, out, grads = run(arm)
            if repeat >= warmup:
                times[arm].append(ms)
            if repeat != warmup:
                continue
            if arm == "default":
                reference = (out, grads)
            else:
                check[arm] = {
                    "outputs_identical": all(
                        torch.equal(out[k], reference[0][k]) for k in out
                    ),
                    "gradient_rel_l2": {
                        k: rel_l2(g, reference[1][k]) for k, g in grads.items()
                    },
                }
            del out, grads
    return times, check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument(
        "--views", type=int, nargs="+", default=[0, 250, 500, 750, 1000, 1250]
    )
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--channels", type=int, nargs="+", default=list(CHANNELS))
    parser.add_argument("--scope", choices=("raster", "render"), default="raster")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
        "cases": [],
    }
    for view in args.views:
        for scale in args.scales:
            viewmat, K, width, height = scaled_camera(cameras[view], scale)
            generator = torch.Generator().manual_seed(2026 + view)
            if args.scope == "raster":
                runs, n_isects = raster_cases(
                    inputs, viewmat, K, width, height, args.channels, generator
                )
                cases = [(c, runs[c], {"intersections": n_isects}) for c in runs]
            else:
                cases = [
                    (8, render_case(inputs, viewmat, K, width, height, generator), {})
                ]
            for channels, run, info in cases:
                times, check, guard = guarded_measure(run, args.warmup, args.repeats)
                if args.scope == "raster":
                    summary = {a: statistics.median(times[a]) for a in ARMS}
                else:
                    summary = {
                        a: {
                            "forward": statistics.median(t[0] for t in times[a]),
                            "backward": statistics.median(t[1] for t in times[a]),
                            "total": statistics.median(t[0] + t[1] for t in times[a]),
                        }
                        for a in ARMS
                    }
                case = {
                    "view": view,
                    "scale": scale,
                    "size": [width, height],
                    "channels": channels,
                    **info,
                    "median_ms": summary,
                    "checks": check,
                    "gpu_guard": guard,
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
            del cases
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
