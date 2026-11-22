"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
def test_rasterization(
    per_view_color: bool, sh_degree: Optional[int], render_mode: str, packed: bool
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 2, 10_000
    means = torch.rand(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    scales = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(C, N, 3, device=device)
        else:
            colors = torch.rand(C, N, (sh_degree + 1) ** 2, 3, device=device)
    else:
        if sh_degree is None:
            colors = torch.rand(N, 3, device=device)
        else:
            colors = torch.rand(N, (sh_degree + 1) ** 2, 3, device=device)

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(C, -1, -1)
    viewmats = torch.eye(4, device=device).expand(C, -1, -1)

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [True, False])
def test_rasterization_blending_stats(packed: bool):
    from gsplat.rendering import rasterization

    torch.manual_seed(7)
    count = 32
    means = torch.randn(count, 3, device=device)
    means[:, 2] += 3
    means.requires_grad_()
    quats = torch.randn(count, 4, device=device)
    scales = torch.full((count, 3), 0.1, device=device)
    opacities = torch.full((count,), 0.5, device=device)
    colors = torch.rand(count, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    renders, alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        32,
        32,
        packed=packed,
        blending_stats=True,
    )
    stats = meta["blending_stats"]
    (renders.sum() + alphas.sum()).backward()

    assert stats.shape[-1] == 2
    assert torch.all(stats >= 0)
    assert torch.all(stats[..., 0] <= stats[..., 1])
    assert torch.any(stats[..., 1] > 0)

    _, _, disabled_meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        32,
        32,
        packed=packed,
    )
    assert "blending_stats" not in disabled_meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterization_blending_stats_exact():
    from gsplat.rendering import rasterization

    means = torch.tensor([[0.0, 0.0, 3.0]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), 0.1, device=device)
    opacities = torch.tensor([0.5], device=device)
    colors = torch.ones((1, 3), device=device)
    viewmats = torch.eye(4, device=device)[None].expand(2, -1, -1).contiguous()
    Ks = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
        device=device,
    ).expand(2, -1, -1).contiguous()

    renders, alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        1,
        1,
        packed=False,
        blending_stats=True,
    )
    (renders.sum() + alphas.sum()).backward()

    torch.testing.assert_close(
        meta["blending_stats"], torch.tensor([[[0.5, 1.0]], [[0.5, 1.0]]], device=device)
    )
