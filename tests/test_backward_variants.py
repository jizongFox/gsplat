"""Backward variants of rasterize_to_pixels (PennyGS, arXiv 2611.27525).

Every variant must reproduce the default per-pixel gradients up to summation
order, including partial edge tiles, backgrounds, tile masks, packed inputs,
absgrad, several cameras and saturated or degenerate Gaussians.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")
VARIANTS = ["pixel_lean"]


def rel_l2(value, ref):
    return float(
        (value.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )


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
    width = width or w
    height = height or h
    torch.manual_seed(7)
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
    if packed:
        camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = projected
    else:
        radii, means2d, depths, conics, _ = projected
    tile_size = 16
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    if packed:
        colors = torch.randn(len(means2d), channels, device=device)
        opac = opacities[gaussian_ids]
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tw,
            th,
            packed=True,
            n_cameras=C,
            camera_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        colors = torch.randn(C, len(means), channels, device=device)
        opac = opacities.repeat(C, 1)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tw, th
        )
    offsets = isect_offset_encode(isect_ids, C, tw, th)
    backgrounds = torch.rand((C, channels), device=device)
    return {
        "inputs": [t.detach().clone() for t in (means2d, conics, colors, opac)],
        "backgrounds": backgrounds,
        "width": width,
        "height": height,
        "offsets": offsets,
        "flatten_ids": flatten_ids,
        "packed": packed,
    }


def _grads(scene, backward, masks=None):
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
        backward=backward,
    )
    torch.manual_seed(11)
    v_image, v_alpha = torch.randn_like(image), torch.randn_like(alpha)
    grads = torch.autograd.grad(
        (image * v_image).sum() + (alpha * v_alpha).sum(), leaves
    )
    return image, alpha, list(grads) + [leaves[0].absgrad]


def _assert_matches_pixel(scene, variant, masks=None):
    image, alpha, reference = _grads(scene, "pixel", masks)
    image_v, alpha_v, grads = _grads(scene, variant, masks)
    if masks is not None:
        # The forward leaves masked tiles unwritten (uninitialized memory).
        keep = masks.repeat_interleave(16, 1).repeat_interleave(16, 2)
        keep = keep[:, : scene["height"], : scene["width"]]
        image, image_v, alpha, alpha_v = (
            t[keep] for t in (image, image_v, alpha, alpha_v)
        )
    assert torch.equal(image, image_v) and torch.equal(alpha, alpha_v)
    for name, grad, ref in zip(
        ("means2d", "conics", "colors", "opacities", "absgrad"), grads, reference
    ):
        assert torch.isfinite(grad).all(), name
        assert rel_l2(grad, ref) <= 1e-4, (variant, name, rel_l2(grad, ref))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("channels", [1, 3, 4, 7, 8])
def test_variants_match_pixel_backward(variant, channels):
    # 250x170 leaves partial edge tiles on both axes
    _assert_matches_pixel(_scene(channels, width=250, height=170), variant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_packed_masked_and_stressed(variant):
    _assert_matches_pixel(_scene(7, packed=True), variant)
    scene = _scene(7, stress=True)
    _assert_matches_pixel(scene, variant)
    masks = torch.rand(scene["offsets"].shape, device=device) > 0.3
    _assert_matches_pixel(scene, variant, masks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_variants_match_torch_reference():
    from gsplat.cuda._torch_impl import _rasterize_to_pixels

    scene = _scene(3)
    leaves = [t.clone().requires_grad_() for t in scene["inputs"]]
    image, alpha = _rasterize_to_pixels(
        *leaves,
        scene["width"],
        scene["height"],
        16,
        scene["offsets"],
        scene["flatten_ids"],
        backgrounds=scene["backgrounds"],
    )
    torch.manual_seed(11)
    v_image, v_alpha = torch.randn_like(image), torch.randn_like(alpha)
    reference = torch.autograd.grad(
        (image * v_image).sum() + (alpha * v_alpha).sum(), leaves
    )
    for variant in VARIANTS:
        grads = _grads(scene, variant)[2][:4]
        for grad, ref, tol in zip(grads, reference, (5e-3, 1e-3, 1e-3, 2e-3)):
            torch.testing.assert_close(grad, ref, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_unknown_backward_is_rejected():
    from gsplat.cuda._wrapper import rasterize_to_pixels

    scene = _scene(3)
    with pytest.raises(ValueError, match="Unknown backward"):
        rasterize_to_pixels(
            *scene["inputs"],
            scene["width"],
            scene["height"],
            16,
            scene["offsets"],
            scene["flatten_ids"],
            backward="warp",
        )
