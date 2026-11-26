"""32-bit sort keys for tile binning (PennyGS, arXiv 2611.27525 Sec. 3.3).

Shortened keys must give the same tile ranges and, per tile, the same Gaussians in
depth order up to the kept depth precision.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _projected(packed=False):
    from gsplat.cuda._wrapper import fully_fused_projection

    means, quats, scales, opacities, _, viewmats, Ks, width, height = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    projected = fully_fused_projection(
        means, None, quats, scales * 0.1, viewmats, Ks, width, height, packed=packed
    )
    C = len(Ks)
    if packed:
        camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = projected
        opac = opacities[gaussian_ids]
        extra = dict(
            packed=True, n_cameras=C, camera_ids=camera_ids, gaussian_ids=gaussian_ids
        )
    else:
        radii, means2d, depths, conics, _ = projected
        opac = opacities.repeat(C, 1)
        extra = {}
    return means2d, radii, depths, conics, opac, width, height, C, extra


def _bin(short_keys, projected, tile_size=16):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    means2d, radii, depths, conics, opac, width, height, C, extra = projected
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tw,
        th,
        conics=conics,
        opacities=opac,
        compact_box=True,
        short_keys=short_keys,
        **extra,
    )
    offsets = isect_offset_encode(ids, C, tw, th)
    depth_bits = (
        32 - (math.floor(math.log2(tw * th)) + 1) - (math.floor(math.log2(C)) + 1)
    )
    return ids, flatten_ids, offsets, depths.reshape(-1), depth_bits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [False, True])
def test_short_keys_keep_tiles_and_depth_order(packed):
    projected = _projected(packed)
    ids64, flat64, offsets64, depths, _ = _bin(False, projected)
    ids32, flat32, offsets32, _, depth_bits = _bin(True, projected)
    assert ids32.dtype == torch.int32 and ids64.dtype == torch.int64
    assert torch.equal(offsets32, offsets64)
    assert flat32.numel() == flat64.numel()
    # same Gaussians per tile
    starts = offsets64.reshape(-1).tolist() + [flat64.numel()]
    for a, b in zip(starts[:-1], starts[1:]):
        assert torch.equal(flat64[a:b].sort().values, flat32[a:b].sort().values)
    # depth order within the kept precision: keys are sorted, and every inversion
    # of the exact order is between depths that agree in the kept bits
    keys = ids32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    assert bool((keys[1:] >= keys[:-1]).all())
    d = depths[flat32]
    same_tile = (keys[1:] >> depth_bits) == (keys[:-1] >> depth_bits)
    inversions = same_tile & (d[1:] < d[:-1])
    relative = ((d[:-1] - d[1:]) / d[1:])[inversions]
    assert relative.numel() == 0 or float(relative.max()) <= 2.0 ** -(depth_bits - 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_short_keys_render_like_long_keys():
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, colors, viewmats, Ks, width, height = (
        load_test_data(
            device=device,
            data_path=os.path.join(
                os.path.dirname(__file__), "../assets/test_garden.npz"
            ),
        )
    )
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB+ED",
        compact_box=True,
    )
    image64, alpha64, _ = rasterization(**kwargs)
    image64b, _, _ = rasterization(**kwargs)
    image32, alpha32, meta = rasterization(**kwargs, isect_short_keys=True)
    assert torch.equal(image64, image64b), "fixture must be deterministic"
    assert meta["isect_ids"].dtype == torch.int32
    # Only Gaussians within the kept depth precision change blending order:
    # garden (3 cameras, 19 depth bits) renders at ~50.7 dB PSNR against long keys.
    rgb64, rgb32 = image64[..., :3].clamp(0, 1), image32[..., :3].clamp(0, 1)
    psnr = -10 * torch.log10(((rgb64 - rgb32) ** 2).mean())
    assert float(psnr) > 45.0
    assert float((alpha32 - alpha64).abs().mean()) < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_short_keys_reject_too_many_tiles():
    with pytest.raises(RuntimeError, match="depth bits"):
        _bin(True, _projected(), tile_size=1)
