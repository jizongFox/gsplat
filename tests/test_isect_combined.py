"""32-bit sort keys combined with single-pass binning (PennyGS, arXiv 2611.27525
Sec. 3.3). Speed experiment: the combined path must reproduce the two-pass
short-key output exactly, and without the tie fix-up the same pairs.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _projected(packed):
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
    tw, th = math.ceil(width / 16), math.ceil(height / 16)
    args = (means2d, radii, depths, 16, tw, th)
    kwargs = dict(conics=conics, opacities=opac, compact_box=True, **extra)
    return args, kwargs, C


def _pairs(ids, flatten_ids):
    # canonical order of (key, Gaussian) pairs: equal keys may come in any order
    order = torch.argsort(flatten_ids.to(torch.int64), stable=True)
    order = order[torch.argsort(ids[order].to(torch.int64), stable=True)]
    return ids[order], flatten_ids[order]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [False, True])
def test_combined_matches_two_pass_short_keys(packed):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    args, kwargs, C = _projected(packed)
    tiles_b, ids_b, flat_b = isect_tiles(*args, **kwargs, short_keys=True)
    assert ids_b.dtype == torch.int32
    # short keys have many more runs of equal keys than 64-bit ids
    assert int((ids_b[1:] == ids_b[:-1]).sum()) > 0
    capacity = 2 * ids_b.numel()
    tiles_d, ids_d, flat_d = isect_tiles(
        *args, **kwargs, short_keys=True, single_pass_capacity=capacity
    )
    assert ids_d.dtype == torch.int32
    assert torch.equal(tiles_d, tiles_b)
    assert torch.equal(ids_d, ids_b)
    assert torch.equal(flat_d, flat_b)  # tie fix-up restores the two-pass order
    # without the fix-up: same sorted keys, tile ranges and pairs
    tiles_e, ids_e, flat_e = isect_tiles(
        *args,
        **kwargs,
        short_keys=True,
        single_pass_capacity=capacity,
        single_pass_restore_ties=False,
    )
    assert torch.equal(tiles_e, tiles_b) and torch.equal(ids_e, ids_b)
    tw, th = args[4], args[5]
    assert torch.equal(
        isect_offset_encode(ids_e, C, tw, th), isect_offset_encode(ids_b, C, tw, th)
    )
    for a, b in zip(_pairs(ids_e, flat_e), _pairs(ids_b, flat_b)):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_combined_falls_back_above_capacity(capfd):
    from gsplat.cuda._wrapper import isect_tiles

    args, kwargs, _ = _projected(False)
    _, ids_b, flat_b = isect_tiles(*args, **kwargs, short_keys=True)
    _, ids_d, flat_d = isect_tiles(
        *args, **kwargs, short_keys=True, single_pass_capacity=ids_b.numel() // 2
    )
    assert "above its capacity" in capfd.readouterr().err  # TORCH_WARN -> stderr
    assert torch.equal(ids_d, ids_b) and torch.equal(flat_d, flat_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_combined_renders_like_two_pass_short_keys():
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
        isect_short_keys=True,
    )
    image_b, alpha_b, meta_b = rasterization(**kwargs)
    image_d, alpha_d, meta_d = rasterization(
        **kwargs, isect_capacity=2 * meta_b["isect_ids"].numel()
    )
    assert torch.equal(meta_d["flatten_ids"], meta_b["flatten_ids"])
    assert torch.equal(image_d, image_b) and torch.equal(alpha_d, alpha_b)
