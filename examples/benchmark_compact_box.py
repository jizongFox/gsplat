"""Compact Box A/B benchmark with synthetic Gaussians.

Example:
    python examples/benchmark_compact_box.py --packed False --iters 20 --warmup 5
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

import torch
import torch.nn.functional as F
import tyro

from gsplat.rendering import rasterization


class ModeResult(TypedDict):
    render: torch.Tensor
    meta: Dict[str, Any]
    avg_ms: float
    n_isects: int
    tiles_mean: float


@dataclass
class Config:
    seed: int = 42
    gaussians: int = 120000
    cameras: int = 2
    width: int = 1280
    height: int = 720
    warmup: int = 10
    iters: int = 50
    packed: bool = False
    compact_box_mult: float = 1.0
    compact_box_tau2: Optional[float] = None
    device: str = "cuda"


def make_synthetic_scene(
    device: torch.device,
    n_gaussians: int,
    n_cameras: int,
    width: int,
    height: int,
) -> Dict[str, torch.Tensor]:
    means = torch.empty(n_gaussians, 3, device=device)
    means[:, 0] = torch.randn(n_gaussians, device=device) * 1.2
    means[:, 1] = torch.randn(n_gaussians, device=device) * 0.8
    means[:, 2] = torch.rand(n_gaussians, device=device) * 3.0 + 1.0

    quats = F.normalize(torch.randn(n_gaussians, 4, device=device), dim=-1)
    scales = torch.rand(n_gaussians, 3, device=device) * 0.08 + 0.01
    opacities = torch.sigmoid(torch.randn(n_gaussians, device=device) * 0.5)
    colors = torch.rand(n_gaussians, 3, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(n_cameras, 1, 1)
    if n_cameras > 1:
        # Small x-axis camera shifts to create multi-view variation.
        offsets = torch.linspace(-0.08, 0.08, steps=n_cameras, device=device)
        viewmats[:, 0, 3] = -offsets

    focal = 0.9 * float(max(width, height))
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    Ks = Ks.repeat(n_cameras, 1, 1)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
    }


@torch.no_grad()
def run_mode(
    scene: Dict[str, torch.Tensor],
    width: int,
    height: int,
    packed: bool,
    warmup: int,
    iters: int,
    compact_box: bool,
    compact_box_mult: float,
    compact_box_tau2: Optional[float],
) -> ModeResult:
    last_render = None
    last_meta = None

    if scene["means"].is_cuda:
        torch.cuda.synchronize(scene["means"].device)

    raster_kwargs: Dict[str, Any] = {
        "means": scene["means"],
        "quats": scene["quats"],
        "scales": scene["scales"],
        "opacities": scene["opacities"],
        "colors": scene["colors"],
        "viewmats": scene["viewmats"],
        "Ks": scene["Ks"],
        "width": width,
        "height": height,
        "packed": packed,
        "compact_box": compact_box,
        "compact_box_mult": compact_box_mult,
        "compact_box_tau2": compact_box_tau2,
    }

    for _ in range(warmup):
        last_render, _, last_meta = rasterization(**raster_kwargs)

    if scene["means"].is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            last_render, _, last_meta = rasterization(**raster_kwargs)
        end.record()
        torch.cuda.synchronize(scene["means"].device)
        avg_ms = start.elapsed_time(end) / float(iters)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            last_render, _, last_meta = rasterization(**raster_kwargs)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / float(iters)

    assert last_render is not None and last_meta is not None

    tiles_per_gauss = last_meta["tiles_per_gauss"].float()
    n_isects = int(last_meta["isect_ids"].numel())
    out: ModeResult = {
        "render": last_render,
        "meta": last_meta,
        "avg_ms": avg_ms,
        "n_isects": n_isects,
        "tiles_mean": float(tiles_per_gauss.mean().item()),
    }
    return out


def image_diff_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    mse = float(((a - b) ** 2).mean().item())
    rmse = math.sqrt(max(mse, 0.0))
    if mse <= 1e-20:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(1.0 / mse)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
    }


def format_float(x: float) -> str:
    if math.isinf(x):
        return "inf"
    return f"{x:.6f}"


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device=cuda was requested")

    torch.manual_seed(cfg.seed)
    scene = make_synthetic_scene(
        device=device,
        n_gaussians=cfg.gaussians,
        n_cameras=cfg.cameras,
        width=cfg.width,
        height=cfg.height,
    )

    print("=== Compact Box A/B Benchmark ===")
    print(
        f"device={device} gaussians={cfg.gaussians} cameras={cfg.cameras} "
        f"resolution={cfg.width}x{cfg.height} packed={cfg.packed} tile_size=16"
    )

    base = run_mode(
        scene=scene,
        width=cfg.width,
        height=cfg.height,
        packed=cfg.packed,
        warmup=cfg.warmup,
        iters=cfg.iters,
        compact_box=False,
        compact_box_mult=1.0,
        compact_box_tau2=None,
    )
    cb = run_mode(
        scene=scene,
        width=cfg.width,
        height=cfg.height,
        packed=cfg.packed,
        warmup=cfg.warmup,
        iters=cfg.iters,
        compact_box=True,
        compact_box_mult=cfg.compact_box_mult,
        compact_box_tau2=cfg.compact_box_tau2,
    )

    diff = image_diff_metrics(base["render"], cb["render"])

    print("\n[Baseline]")
    print(
        f"avg_render_ms={base['avg_ms']:.3f} n_isects={base['n_isects']} "
        f"tiles_per_gauss_mean={base['tiles_mean']:.4f}"
    )

    print("\n[CompactBox]")
    tau2_display = cfg.compact_box_tau2
    if tau2_display is None:
        tau2_display = 9.0 * cfg.compact_box_mult * cfg.compact_box_mult
    print(
        f"avg_render_ms={cb['avg_ms']:.3f} n_isects={cb['n_isects']} "
        f"tiles_per_gauss_mean={cb['tiles_mean']:.4f} "
        f"mult={cfg.compact_box_mult:.4f} tau2={tau2_display:.6f}"
    )

    isect_delta = cb["n_isects"] - base["n_isects"]
    isect_ratio = 0.0
    if base["n_isects"] > 0:
        isect_ratio = 100.0 * (cb["n_isects"] / base["n_isects"] - 1.0)
    speedup = 0.0
    if cb["avg_ms"] > 0:
        speedup = base["avg_ms"] / cb["avg_ms"]

    print("\n[Delta: CB - Baseline]")
    print(f"n_isects_delta={isect_delta} ({isect_ratio:+.2f}%)")
    print(f"avg_render_ms_speedup={speedup:.4f}x")
    print(
        "image_diff "
        f"max_abs={format_float(diff['max_abs'])} "
        f"mean_abs={format_float(diff['mean_abs'])} "
        f"rmse={format_float(diff['rmse'])} "
        f"psnr={format_float(diff['psnr'])}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
