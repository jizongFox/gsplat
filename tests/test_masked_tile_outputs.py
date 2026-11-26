"""Masked-out tiles render the background with alpha 0 and last_ids 0.

The outputs are allocated uninitialized, so a masked tile used to leave stale
memory in alpha and last_ids; the background gradient (weighted by 1 - alpha)
then depended on that memory. The tests fill the allocator's cached blocks with
NaN first, so stale values would show.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _scene(channels, width=250, height=170):
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
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales * 0.1, viewmats, Ks, width, height
    )
    tw, th = math.ceil(width / 16), math.ceil(height / 16)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, 16, tw, th)
    torch.manual_seed(3)
    masks = torch.rand((C, th, tw), device=device) > 0.4
    masks[:, -1, :] = False  # partial bottom-edge tiles
    masks[:, :, -1] = False  # partial right-edge tiles
    return dict(
        means2d=means2d.detach(),
        conics=conics.detach(),
        colors=torch.rand((C, len(means), channels), device=device),
        opacities=opacities.repeat(C, 1).detach(),
        backgrounds=torch.rand((C, channels), device=device),
        width=width,
        height=height,
        offsets=isect_offset_encode(isect_ids, C, tw, th),
        flatten_ids=flatten_ids,
        masks=masks,
    )


def _pixel_mask(scene):
    keep = scene["masks"].repeat_interleave(16, 1).repeat_interleave(16, 2)
    return ~keep[:, : scene["height"], : scene["width"]]  # True where masked


def _poison_allocator(*shapes):
    """Leave NaN-filled blocks of these sizes in the caching allocator."""
    blocks = [torch.full(s, float("nan"), device=device) for s in shapes]
    del blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("forward", ["pixel", "subtile"])
def test_masked_tiles_have_zero_alpha_and_last_ids(forward):
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    scene = _scene(3)
    C, H, W = len(scene["masks"]), scene["height"], scene["width"]
    masked = _pixel_mask(scene)
    for _ in range(3):
        _poison_allocator((C, H, W, 3), (C, H, W, 1), (C, H, W))
        render, alpha, last_ids = _make_lazy_cuda_func("rasterize_to_pixels_fwd")(
            scene["means2d"].contiguous(),
            scene["conics"].contiguous(),
            scene["colors"].contiguous(),
            scene["opacities"].contiguous(),
            scene["backgrounds"].contiguous(),
            scene["masks"].contiguous(),
            W,
            H,
            16,
            scene["offsets"],
            scene["flatten_ids"],
            forward == "subtile",
        )
        assert bool(masked.any()) and bool((~masked).any())
        assert bool((alpha[..., 0][masked] == 0).all())
        assert bool((last_ids[masked] == 0).all())
        expected = scene["backgrounds"][:, None, None, :].expand_as(render)
        assert torch.equal(render[masked], expected[masked])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("forward", ["pixel", "subtile"])
@pytest.mark.parametrize("channels", [3, 7])
def test_masked_tiles_background_gradient(forward, channels):
    """The background gradient sums v_color * (1 - alpha) with alpha 0 on
    masked pixels, and the unmasked pixels match an unmasked render."""
    from gsplat.cuda._wrapper import rasterize_to_pixels

    scene = _scene(channels)
    C, H, W = len(scene["masks"]), scene["height"], scene["width"]
    masked = _pixel_mask(scene)

    def run(masks):
        backgrounds = scene["backgrounds"].clone().requires_grad_()
        padded = 8 if channels == 7 else channels  # the wrapper pads 7 to 8
        _poison_allocator((C, H, W, padded), (C, H, W, 1), (C, H, W))
        image, alpha = rasterize_to_pixels(
            scene["means2d"],
            scene["conics"],
            scene["colors"],
            scene["opacities"],
            W,
            H,
            16,
            scene["offsets"],
            scene["flatten_ids"],
            backgrounds=backgrounds,
            masks=masks,
            forward=forward,
        )
        torch.manual_seed(9)
        v_image = torch.randn_like(image)
        (image * v_image).sum().backward()
        return image.detach(), alpha.detach(), backgrounds.grad, v_image

    image, alpha, v_bg, v_image = run(scene["masks"])
    assert bool((alpha[..., 0][masked] == 0).all())
    expected = (v_image * (1.0 - alpha)).sum(dim=(1, 2))
    torch.testing.assert_close(v_bg, expected, rtol=1e-5, atol=1e-5)
    assert bool(torch.isfinite(v_bg).all())

    image_full, alpha_full, _, _ = run(None)
    assert torch.equal(image[~masked], image_full[~masked])
    assert torch.equal(alpha[~masked], alpha_full[~masked])
