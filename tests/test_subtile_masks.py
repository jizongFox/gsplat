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


def _run(scene, subtile_masks, masks=None):
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
        subtile_masks=subtile_masks,
    )
    torch.manual_seed(11)
    v_image, v_alpha = torch.randn_like(image), torch.randn_like(alpha)
    grads = torch.autograd.grad(
        (image * v_image).sum() + (alpha * v_alpha).sum(), leaves
    )
    return image, alpha, list(grads) + [leaves[0].absgrad]


def _assert_identical(scene, masks=None):
    image, alpha, grads = _run(scene, False, masks)
    image_s, alpha_s, grads_s = _run(scene, True, masks)
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
            subtile_masks=True,
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
            subtile_masks=subtile,
        )
        for subtile in (False, True)
    ]
    assert float(images[0][1].max()) > 0  # the default kernel blends it
    assert torch.equal(images[0][0], images[1][0])
    assert torch.equal(images[0][1], images[1][1])
