"""PennyGS (arXiv 2611.27525) Sec. 3.7: projection backward + Adam stage on a real scene.

For each view the projected state is computed once; each variant then runs the
stage "projection backward -> mean / quaternion / log-scale update" from the same
upstream gradients (random on visible Gaussians), timed with CUDA events,
variants alternating inside each repeat:

- ``proj_bwd+adam``: `fully_fused_projection` backward kernel, the log-scale
  chain rule, then dense fused `torch.optim.Adam` with float32 moments (the
  parent application's path);
- ``proj_bwd+selective_adam``: the same backward, then gsplat `SelectiveAdam`
  (visible Gaussians only);
- ``fused_fp32`` / ``fused_bf16``: `fused_projection_adam` with float32 or
  bfloat16 first moments (second moments float32).

The cumulative parameter update of 10 steps from identical states and zero
moments is compared (relative L2 per parameter tensor, max over means /
quaternions / log-scales): fused_fp32 against proj_bwd+selective_adam (same
update rule), fused_bf16 against fused_fp32.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "renderer_benchmark", HERE.parent / "reduce_scatter8" / "renderer_benchmark.py"
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
LRS = (1.6e-4, 1e-3, 5e-3)
VARIANTS = ["proj_bwd+adam", "proj_bwd+selective_adam", "fused_fp32", "fused_bf16"]


def make_case(inputs, viewmat, K, width, height, generator):
    from gsplat.cuda._wrapper import (
        _make_lazy_cuda_func,
        fully_fused_projection,
        fused_projection_adam,
    )
    from gsplat.optimizers import SelectiveAdam

    base = [
        inputs["means"].clone(),
        rb.quaternion_normalize(inputs["quats"]).clone(),
        inputs["log_scales"].clone(),
    ]
    with torch.no_grad():
        radii, _, _, conics, _ = fully_fused_projection(
            base[0],
            None,
            base[1],
            base[2].exp(),
            viewmat[None],
            K[None],
            width,
            height,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1000.0,
        )
    n = len(base[0])
    upstream = [
        torch.randn(shape, generator=generator).cuda()
        for shape in ((1, n, 2), (1, n), (1, n, 3))
    ]
    visible = radii[0] > 0
    proj_bwd = _make_lazy_cuda_func("fully_fused_projection_bwd")
    states = {}

    def fresh(variant):
        params = [torch.nn.Parameter(p.clone()) for p in base]
        if variant == "proj_bwd+adam":
            opt = torch.optim.Adam(
                [{"params": [p], "lr": lr} for p, lr in zip(params, LRS)],
                eps=1e-15,
                fused=True,
            )
        elif variant == "proj_bwd+selective_adam":
            opt = SelectiveAdam(
                [{"params": [p], "lr": lr} for p, lr in zip(params, LRS)],
                eps=1e-15,
                betas=(0.9, 0.999),
            )
        else:
            # compact first moment in bf16; second moment always float32
            dtype = torch.bfloat16 if variant == "fused_bf16" else torch.float32
            opt = [
                torch.zeros_like(p, dtype=dtype if k == 0 else torch.float32)
                for p in params
                for k in range(2)
            ]
        return params, opt

    def backward(params):
        means, quats, log_scales = params
        scales = log_scales.exp()
        v_means, _, v_quats, v_scales, _ = proj_bwd(
            means,
            None,
            quats,
            scales,
            viewmat[None],
            K[None],
            width,
            height,
            0.3,
            gsplat_camera_model(),
            radii,
            conics,
            None,
            *upstream,
            None,
            False,
        )
        means.grad, quats.grad, log_scales.grad = v_means, v_quats, v_scales * scales

    def run(variant):
        if variant not in states:
            states[variant] = fresh(variant)
        params, opt = states[variant]
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        if variant.startswith("fused"):
            with torch.no_grad():
                fused_projection_adam(
                    *params,
                    viewmat,
                    K,
                    width,
                    height,
                    radii[0],
                    conics[0],
                    upstream[0][0],
                    upstream[1][0],
                    upstream[2][0],
                    moments=tuple(opt),
                    lrs=LRS,
                    eps=1e-15,
                )
        else:
            backward(params)
            if variant == "proj_bwd+adam":
                opt.step()
            else:
                opt.step(visible)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def one_step(variant, steps=10):
        """Cumulative parameter update of `steps` steps (same upstream gradients)
        from the shared starting state and zero moments."""
        states.pop(variant, None)
        for _ in range(steps):
            run(variant)
        return [p.detach() - b for p, b in zip(states.pop(variant)[0], base)]

    return run, one_step, int(visible.sum())


def gsplat_camera_model():
    from gsplat.cuda._wrapper import _make_lazy_cuda_obj

    return _make_lazy_cuda_obj("CameraModelType.PINHOLE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument(
        "--views", type=int, nargs="+", default=[0, 250, 500, 750, 1000, 1250]
    )
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
        "views": [],
    }
    for view in args.views:
        camera = cameras[view]
        viewmat, K = (t.cuda().contiguous() for t in rb.camera_tensors(camera))
        width, height = int(camera["width"]), int(camera["height"])
        run, one_step, visible = make_case(
            inputs,
            viewmat,
            K,
            width,
            height,
            torch.Generator().manual_seed(2026 + view),
        )
        times = {v: [] for v in VARIANTS}
        for repeat in range(args.warmup + args.repeats):
            for variant in VARIANTS:
                ms = run(variant)
                if repeat >= args.warmup:
                    times[variant].append(ms)
        selective, fused32, fused16 = (one_step(v) for v in VARIANTS[1:])

        def rel(a, b):
            return max(
                float((x - y).norm() / y.norm().clamp_min(1e-30)) for x, y in zip(a, b)
            )

        report["views"].append(
            {
                "view": view,
                "image": camera["img_name"],
                "visible": visible,
                "median_ms": {v: statistics.median(times[v]) for v in VARIANTS},
                "update_rel_l2_fused_fp32_vs_selective_adam": rel(fused32, selective),
                "update_rel_l2_fused_bf16_vs_fused_fp32": rel(fused16, fused32),
            }
        )
        print(json.dumps(report["views"][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
