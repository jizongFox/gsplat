"""Backward kernel option of `rasterize_to_pixels(backward=...)`.

"block" (default) and "warp_private" (PR #13) run the same per-pixel recurrence
and reductions; only the staging and the order of the float atomics differ, so
gradients agree up to that order. "warp_private" falls back to "block" above 8
channels.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _scene(channels, packed, width=250, height=170):
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
    )

    means, quats, scales, opacities, _, viewmats, Ks, _, _ = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    C = len(Ks)
    projected = fully_fused_projection(
        means, None, quats, scales * 0.1, viewmats, Ks, width, height, packed=packed
    )
    tw, th = math.ceil(width / 16), math.ceil(height / 16)
    g = torch.Generator(device=device).manual_seed(5)
    if packed:
        camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = projected
        colors = torch.rand(len(means2d), channels, generator=g, device=device)
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
        colors = torch.rand(C, len(means), channels, generator=g, device=device)
        opac = opacities.repeat(C, 1)
        _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, 16, tw, th)
    return dict(
        inputs=[t.detach().clone() for t in (means2d, conics, colors, opac)],
        backgrounds=torch.rand((C, channels), generator=g, device=device),
        width=width,
        height=height,
        packed=packed,
        offsets=isect_offset_encode(isect_ids, C, tw, th),
        flatten_ids=flatten_ids,
    )


def _grads(scene, backward):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    leaves = [t.clone().requires_grad_() for t in scene["inputs"]]
    backgrounds = scene["backgrounds"].clone().requires_grad_()
    image, alpha = rasterize_to_pixels(
        *leaves,
        scene["width"],
        scene["height"],
        16,
        scene["offsets"],
        scene["flatten_ids"],
        backgrounds=backgrounds,
        packed=scene["packed"],
        absgrad=True,
        backward=backward,
    )
    torch.manual_seed(13)
    loss = (image * torch.randn_like(image)).sum() + (
        alpha * torch.randn_like(alpha)
    ).sum()
    grads = torch.autograd.grad(loss, leaves + [backgrounds])
    return [image.detach(), alpha.detach()], list(grads) + [leaves[0].absgrad]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4, 7, 8, 9, 16])
def test_warp_private_backward_matches_block(channels, packed):
    scene = _scene(channels, packed)
    outs_b, grads_b = _grads(scene, "block")
    outs_w, grads_w = _grads(scene, "warp_private")
    for a, b in zip(outs_w, outs_b):
        assert torch.equal(a, b)
    for a, b in zip(grads_w, grads_b):
        # only the order of the float atomics differs (two "block" runs
        # scatter the same way), so the tolerance follows the scale
        assert float(b.abs().max()) > 0
        scale = float(b.abs().max())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5 * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterization_passes_the_backward_kernel():
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, colors, viewmats, Ks, _, _ = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )

    def run(backward):
        leaves = [t.clone().requires_grad_() for t in (means, quats, scales, colors)]
        images, alphas, _ = rasterization(
            leaves[0],
            leaves[1],
            leaves[2] * 0.1,
            opacities,
            leaves[3],
            viewmats,
            Ks,
            250,
            170,
            render_mode="RGB+ED",
            rasterize_backward=backward,
        )
        torch.manual_seed(13)
        loss = (images * torch.randn_like(images)).sum()
        return images.detach(), torch.autograd.grad(loss, leaves)

    images_b, grads_b = run("block")
    images_w, grads_w = run("warp_private")
    assert torch.equal(images_w, images_b)
    for a, b in zip(grads_w, grads_b):
        scale = float(b.abs().max())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5 * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_unknown_backward_kernel():
    from gsplat.cuda._wrapper import rasterize_to_pixels

    scene = _scene(3, False)
    with pytest.raises(ValueError, match="backward kernel"):
        rasterize_to_pixels(
            *scene["inputs"],
            scene["width"],
            scene["height"],
            16,
            scene["offsets"],
            scene["flatten_ids"],
            backward="warp",
        )
