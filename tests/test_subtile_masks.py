"""Forward rasterization with subtile masks (PennyGS, arXiv 2611.27525 Sec. 3.4).

The subtile path skips only Gaussians that cannot reach a warp's 8x4 subtile, so
images, alphas and (through last_ids) the backward gradients must be identical.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _scene(channels, *, packed=False, width=None, height=None, stress=False):
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
    )

    means, quats, scales, opacities, _, viewmats, Ks, w, h = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    width, height = width or w, height or h
    torch.manual_seed(5)
    scales = scales * 0.1
    if stress:
        n = len(means)
        opacities = opacities.clone()
        opacities[: n // 8] = 0.9999  # alpha clamps at 0.999
        scales = scales.clone()
        scales[n // 8 : n // 4, 0] *= 1e-4  # needle-like, near-degenerate conics
    C = len(Ks)
    projected = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height, packed=packed
    )
    tw, th = math.ceil(width / 16), math.ceil(height / 16)
    if packed:
        camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = projected
        colors = torch.randn(len(means2d), channels, device=device)
        opac = opacities[gaussian_ids]
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            16,
            tw,
            th,
            packed=True,
            n_cameras=C,
            camera_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        radii, means2d, depths, conics, _ = projected
        colors = torch.randn(C, len(means), channels, device=device)
        opac = opacities.repeat(C, 1)
        _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, 16, tw, th)
    return dict(
        inputs=[t.detach().clone() for t in (means2d, conics, colors, opac)],
        backgrounds=torch.rand((C, channels), device=device),
        width=width,
        height=height,
        packed=packed,
        offsets=isect_offset_encode(isect_ids, C, tw, th),
        flatten_ids=flatten_ids,
    )


def _run(scene, forward, masks=None):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    leaves = [t.clone().requires_grad_() for t in scene["inputs"]]
    image, alpha = rasterize_to_pixels(
        *leaves,
        scene["width"],
        scene["height"],
        16,
        scene["offsets"],
        scene["flatten_ids"],
        backgrounds=scene["backgrounds"],
        masks=masks,
        packed=scene["packed"],
        absgrad=True,
        forward=forward,
    )
    torch.manual_seed(11)
    v_image, v_alpha = torch.randn_like(image), torch.randn_like(alpha)
    grads = torch.autograd.grad(
        (image * v_image).sum() + (alpha * v_alpha).sum(), leaves
    )
    return image, alpha, list(grads) + [leaves[0].absgrad]


def _assert_identical(scene, masks=None):
    image, alpha, grads = _run(scene, "pixel", masks)
    image_s, alpha_s, grads_s = _run(scene, "subtile", masks)
    if masks is not None:
        keep = masks.repeat_interleave(16, 1).repeat_interleave(16, 2)
        keep = keep[:, : scene["height"], : scene["width"]]
        image, image_s, alpha, alpha_s = (
            t[keep] for t in (image, image_s, alpha, alpha_s)
        )
    assert torch.equal(image, image_s) and torch.equal(alpha, alpha_s)
    for g, g_s in zip(grads, grads_s):
        # same last_ids and recurrence; only the atomics' order differs
        torch.testing.assert_close(g_s, g, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [1, 3, 4, 7, 8, 16])
def test_subtile_forward_is_identical(channels):
    # 250x170 leaves partial edge tiles on both axes
    _assert_identical(_scene(channels, width=250, height=170))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_subtile_forward_packed_masked_and_stressed():
    _assert_identical(_scene(7, packed=True))
    scene = _scene(7, stress=True)
    _assert_identical(scene)
    masks = torch.rand(scene["offsets"].shape, device=device) > 0.3
    _assert_identical(scene, masks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [2, 32, 33, 64])
def test_subtile_forward_is_identical_wide(channels):
    # 1-4 channels are staged in shared memory, wider ones read per hit
    _assert_identical(_scene(channels, width=250, height=170))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_subtile_staged_colours_packed_masked_and_stressed():
    _assert_identical(_scene(4, packed=True))
    scene = _scene(3, stress=True)
    _assert_identical(scene)
    masks = torch.rand(scene["offsets"].shape, device=device) > 0.3
    _assert_identical(scene, masks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 7])
def test_subtile_forward_masked_partial_warps(channels):
    # 250x170 edge tiles hold warps that are partly outside the image
    scene = _scene(channels, width=250, height=170)
    masks = torch.rand(scene["offsets"].shape, device=device) > 0.3
    masks[:, -1, :] = False  # every bottom-edge tile is masked
    masks[:, :, -1] = False  # every right-edge tile is masked
    _assert_identical(scene, masks)


def _raw_forward(means2d, conics, colors, opacities, forward):
    """One 16x16 tile; returns render, alpha and last_ids of the CUDA op."""
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    n = means2d.shape[1]
    return _make_lazy_cuda_func("rasterize_to_pixels_fwd")(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        None,
        None,
        16,
        16,
        16,
        torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        torch.arange(n, dtype=torch.int32, device=device),
        forward == "subtile",
    )


def _random_tile_gaussians(seed, n=4096):
    """Gaussians stressing the subtile test: opacities just above 1/255, rotated
    needles up to the ill-conditioning cut, means on subtile edges and far away."""
    g = torch.Generator().manual_seed(seed)
    u = lambda *shape: torch.rand(*shape, generator=g)
    kind = torch.randint(0, 3, (n,), generator=g)
    near = torch.stack((u(n) * 64 - 24, u(n) * 64 - 24), -1)
    edge = (
        torch.stack(
            (
                torch.tensor([0.0, 8.0, 16.0])[torch.randint(0, 3, (n,), generator=g)],
                torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0])[
                    torch.randint(0, 5, (n,), generator=g)
                ],
            ),
            -1,
        )
        + (u(n, 2) - 0.5) * 1e-2
    )
    far = (u(n, 2) - 0.5) * 2e4
    means = torch.where((kind == 0)[:, None], near, edge)
    means = torch.where((kind == 2)[:, None], far, means)
    # conic = R diag(1 / s1^2, 1 / s2^2) R^T with anisotropy up to 1e3
    theta = u(n) * math.pi
    s1 = 0.3 * 10 ** (u(n) * 2)
    s2 = s1 * 10 ** (u(n) * 3)
    a, b = 1 / s1**2, 1 / s2**2
    c, s = torch.cos(theta), torch.sin(theta)
    conics = torch.stack(
        (a * c * c + b * s * s, (a - b) * c * s, a * s * s + b * c * c), -1
    )
    opacities = torch.where(
        u(n) < 0.7, (1 + (u(n) - 0.2) * 0.05) / 255, u(n) * 0.5
    ).clamp(max=0.999)
    colors = u(n, 3)
    to = lambda t: t[None].to(device)
    return to(means), to(conics), to(colors), to(opacities)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("seed", range(8))
def test_subtile_forward_random_single_tile(seed):
    """Exact render, alpha and last_ids against the default kernel."""
    inputs = _random_tile_gaussians(seed)
    ref = _raw_forward(*inputs, "pixel")
    out = _raw_forward(*inputs, "subtile")
    assert float(ref[1].max()) > 0  # the default kernel blends something
    for a, b in zip(out, ref):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_subtile_forward_nonfinite_inputs():
    means, conics, colors, opacities = _random_tile_gaussians(100, n=1024)
    means[0, 0::97, 0] = float("nan")
    conics[0, 1::89, 1] = float("inf")
    conics[0, 2::83, 0] = float("nan")
    opacities[0, 3::79] = float("nan")
    opacities[0, 4::73] = float("inf")
    ref = _raw_forward(means, conics, colors, opacities, "pixel")
    out = _raw_forward(means, conics, colors, opacities, "subtile")
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_unknown_forward_kernel_is_rejected():
    from gsplat.cuda._wrapper import rasterize_to_pixels

    scene = _scene(3)
    means2d, conics, colors, opac = scene["inputs"]
    with pytest.raises(ValueError, match="forward kernel"):
        rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opac,
            scene["width"],
            scene["height"],
            16,
            scene["offsets"],
            scene["flatten_ids"],
            forward="subtile_masks",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_subtile_forward_needs_tile_16():
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
    )

    scene = _scene(3)
    means2d, conics, colors, opac = scene["inputs"]
    with pytest.raises(RuntimeError, match="tile_size 16"):
        rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opac,
            scene["width"],
            scene["height"],
            8,
            torch.zeros(
                (
                    len(colors),
                    math.ceil(scene["height"] / 8),
                    math.ceil(scene["width"] / 8),
                ),
                dtype=torch.int32,
                device=device,
            ),
            scene["flatten_ids"],
            forward="subtile",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_subtile_masks_cover_float32_cancellation():
    """A far, very anisotropic Gaussian whose float32 sigma at pixel (0, 0) is
    accepted by the renderer although the exact ellipse misses the tile."""
    from gsplat.cuda._wrapper import rasterize_to_pixels

    means2d = torch.tensor([[[-9331.5, 10250.751953125]]], device=device)
    conics = torch.tensor([[[0.652995348, 0.594497681, 0.541240573]]], device=device)
    colors = torch.ones((1, 1, 3), device=device)
    opacities = torch.ones((1, 1), device=device)
    offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.zeros(1, dtype=torch.int32, device=device)
    images = [
        rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            16,
            16,
            16,
            offsets,
            flatten_ids,
            forward=forward,
        )
        for forward in ("pixel", "subtile")
    ]
    assert float(images[0][1].max()) > 0  # the default kernel blends it
    assert torch.equal(images[0][0], images[1][0])
    assert torch.equal(images[0][1], images[1][1])
