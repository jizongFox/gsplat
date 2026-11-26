"""Single-pass tile binning (PennyGS, arXiv 2611.27525 Sec. 3.3).

With enough capacity the single pass must produce the same counts, keys and
(per key) Gaussians as the two-pass default; above the capacity it falls back.
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
    return (means2d, radii, depths, 16, tw, th), dict(
        conics=conics, opacities=opac, compact_box=True, **extra
    )


def _pairs(ids, flatten_ids):
    # canonical order of (key, Gaussian) pairs: equal keys may come in any order
    order = torch.argsort(flatten_ids.to(torch.int64), stable=True)
    order = order[torch.argsort(ids[order], stable=True)]
    return ids[order], flatten_ids[order]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [False, True])
def test_single_pass_matches_two_passes(packed):
    from gsplat.cuda._wrapper import isect_tiles

    args, kwargs = _projected(packed)
    tiles2, ids2, flat2 = isect_tiles(*args, **kwargs)
    tiles1, ids1, flat1 = isect_tiles(
        *args, **kwargs, single_pass_capacity=2 * ids2.numel()
    )
    assert torch.equal(tiles1, tiles2)
    assert torch.equal(ids1, ids2)  # sorted keys
    assert torch.equal(flat1, flat2)  # equal keys in Gaussian index order
    for a, b in zip(_pairs(ids1, flat1), _pairs(ids2, flat2)):
        assert torch.equal(a, b)
    # unsorted output holds the same pairs too
    _, ids_u, flat_u = isect_tiles(
        *args, **kwargs, sort=False, single_pass_capacity=ids2.numel()
    )
    for a, b in zip(_pairs(ids_u, flat_u), _pairs(ids2, flat2)):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_single_pass_falls_back_above_capacity(capfd):
    from gsplat.cuda._wrapper import isect_tiles

    args, kwargs = _projected(False)
    _, ids2, flat2 = isect_tiles(*args, **kwargs)
    _, ids1, flat1 = isect_tiles(
        *args, **kwargs, single_pass_capacity=ids2.numel() // 2
    )
    assert "above its capacity" in capfd.readouterr().err  # TORCH_WARN -> stderr
    assert torch.equal(ids1, ids2) and torch.equal(flat1, flat2)


def _scene():
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = (
        load_test_data(
            device=device,
            data_path=os.path.join(
                os.path.dirname(__file__), "../assets/test_garden.npz"
            ),
        )
    )
    return dict(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_single_pass_renders_like_two_passes():
    from gsplat.rendering import rasterization

    # including runs of exactly equal keys (co-located Gaussians, float-equal
    # depths), which are put back in Gaussian index order
    kwargs = _scene()
    image2, alpha2, meta = rasterization(**kwargs)
    ids = meta["isect_ids"]
    assert int((ids[1:] == ids[:-1]).sum()) > 0  # the scene does have ties
    image1, alpha1, meta1 = rasterization(**kwargs, isect_capacity=2 * ids.numel())
    assert torch.equal(meta1["flatten_ids"], meta["flatten_ids"])
    assert torch.equal(image1, image2) and torch.equal(alpha1, alpha2)
