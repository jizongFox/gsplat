"""2DGS forward with subtile masks (`rasterize_to_pixels_2dgs(forward="subtile")`).

The subtile kernel skips only surfels that cannot reach a warp's 8x4 subtile and
shares the default kernel's per-pixel step and blending, so all seven outputs
(colours, alphas, normals, distortion, median depth, last_ids, median_ids) must be
identical, and the backward gradients equal up to the order of float atomics.
Masked-out tiles render empty pixels in both kernels.
"""

import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")
OUTPUTS = ("colors", "alphas", "normals", "distort", "median", "last_ids", "median_ids")


def _raw(
    means2d,
    rt,
    colors,
    opacities,
    normals,
    width,
    height,
    offsets,
    ids,
    forward,
    backgrounds=None,
    masks=None,
):
    """The CUDA op: all seven outputs."""
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    return _make_lazy_cuda_func("rasterize_to_pixels_fwd_2dgs")(
        means2d.contiguous(),
        rt.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        None if backgrounds is None else backgrounds.contiguous(),
        None if masks is None else masks.contiguous(),
        width,
        height,
        16,
        offsets,
        ids,
        forward == "subtile",
    )


def _assert_same(out, ref):
    for name, a, b in zip(OUTPUTS, out, ref):
        assert torch.equal(a, b) or torch.allclose(
            a, b, rtol=0, atol=0, equal_nan=True
        ), name


def _garden(channels, width=250, height=170, packed=False):
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        isect_tiles,
    )

    means, quats, scales, opacities, _, viewmats, Ks, _, _ = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    C = len(Ks)
    proj = fully_fused_projection_2dgs(
        means, quats, scales * 0.1, viewmats, Ks, width, height, packed=packed
    )
    tw, th = math.ceil(width / 16), math.ceil(height / 16)
    g = torch.Generator(device=device).manual_seed(4)
    if packed:
        camera_ids, gaussian_ids, radii, means2d, depths, rt, normals = proj
        n = len(means2d)
        _, isect_ids, ids = isect_tiles(
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
        opac = opacities[gaussian_ids]
        lead = (n,)
    else:
        radii, means2d, depths, rt, normals = proj
        _, isect_ids, ids = isect_tiles(means2d, radii, depths, 16, tw, th)
        opac = opacities.repeat(C, 1)
        lead = (C, len(means))
    # the last colour channel is the depth (2DGS distortion and median read it)
    colors = torch.rand(*lead, channels, generator=g, device=device)
    colors[..., -1] = depths
    return dict(
        means2d=means2d.detach(),
        rt=rt.detach(),
        colors=colors,
        opacities=opac.detach(),
        normals=normals.detach(),
        backgrounds=torch.rand((C, channels), generator=g, device=device),
        width=width,
        height=height,
        offsets=isect_offset_encode(isect_ids, C, tw, th),
        ids=ids,
    )


def _raw_scene(s, forward, masks=None, backgrounds=True):
    return _raw(
        s["means2d"],
        s["rt"],
        s["colors"],
        s["opacities"],
        s["normals"],
        s["width"],
        s["height"],
        s["offsets"],
        s["ids"],
        forward,
        s["backgrounds"] if backgrounds else None,
        masks,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 8, 16, 32, 33, 64])
def test_2dgs_subtile_forward_is_identical(channels):
    # 250x170 leaves partial edge tiles; 1-4 channels are staged in shared memory
    s = _garden(channels)
    for backgrounds in (True, False):
        _assert_same(
            _raw_scene(s, "subtile", backgrounds=backgrounds),
            _raw_scene(s, "pixel", backgrounds=backgrounds),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_2dgs_subtile_forward_packed():
    s = _garden(4, packed=True)
    _assert_same(_raw_scene(s, "subtile"), _raw_scene(s, "pixel"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [4, 8])
def test_2dgs_subtile_backward_matches(channels):
    from gsplat.cuda._wrapper import rasterize_to_pixels_2dgs

    s = _garden(channels)

    def run(forward):
        leaves = {
            k: s[k].clone().requires_grad_()
            for k in ("means2d", "rt", "colors", "opacities", "normals", "backgrounds")
        }
        densify = torch.zeros_like(s["means2d"], requires_grad=True)
        outs = rasterize_to_pixels_2dgs(
            leaves["means2d"],
            leaves["rt"],
            leaves["colors"],
            leaves["opacities"],
            leaves["normals"],
            densify,
            s["width"],
            s["height"],
            16,
            s["offsets"],
            s["ids"],
            backgrounds=leaves["backgrounds"],
            absgrad=True,
            distloss=True,
            forward=forward,
        )
        torch.manual_seed(13)
        loss = sum((o * torch.randn_like(o)).sum() for o in outs)
        grads = torch.autograd.grad(loss, list(leaves.values()) + [densify])
        return [o.detach() for o in outs], list(grads) + [leaves["means2d"].absgrad]

    outs_p, grads_p = run("pixel")
    outs_s, grads_s = run("subtile")
    for a, b in zip(outs_s, outs_p):
        assert torch.equal(a, b)
    for a, b in zip(grads_s, grads_p):
        # same forward outputs and recurrence; only the order of float atomics
        # differs (as between two "pixel" runs), so the tolerance follows the scale
        scale = float(b.abs().max())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5 * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("packed", [False, True])
def test_rasterization_2dgs_subtile_forward(packed):
    from gsplat.rendering import rasterization_2dgs

    means, quats, scales, opacities, colors, viewmats, Ks, _, _ = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    # RGB backgrounds; rasterization_2dgs appends 0 for the depth channel
    backgrounds = torch.rand((len(Ks), 3), device=device)

    def run(forward):
        leaves = [t.clone().requires_grad_() for t in (means, quats, scales, colors)]
        outs = rasterization_2dgs(
            *leaves[:3],
            opacities,
            leaves[3],
            viewmats,
            Ks,
            250,
            170,
            packed=packed,
            backgrounds=backgrounds,
            render_mode="RGB+ED",
            distloss=True,
            depth_mode="median",
            rasterize_forward=forward,
        )
        images = [o for o in outs[:6] if o is not None]
        torch.manual_seed(13)
        loss = sum((o * torch.randn_like(o)).sum() for o in images)
        return [o.detach() for o in images], torch.autograd.grad(loss, leaves)

    images_p, grads_p = run("pixel")
    images_s, grads_s = run("subtile")
    for a, b in zip(images_s, images_p):
        assert torch.equal(a, b)
    for a, b in zip(grads_s, grads_p):
        # the order of float atomics in the backward differs run to run (two
        # "pixel" runs scatter the same way), so the tolerance follows the scale
        scale = float(b.abs().max())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5 * scale)


def _poison_allocator(*shapes):
    """Leave NaN-filled blocks of these sizes in the caching allocator."""
    blocks = [torch.full(sh, float("nan"), device=device) for sh in shapes]
    del blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("forward", ["pixel", "subtile"])
@pytest.mark.parametrize("channels", [4, 8])
def test_2dgs_masked_tiles_render_empty_pixels(forward, channels):
    s = _garden(channels)
    C, H, W = len(s["offsets"]), s["height"], s["width"]
    torch.manual_seed(3)
    masks = torch.rand(s["offsets"].shape, device=device) > 0.4
    masks[:, -1, :] = False  # partial bottom-edge tiles: warps partly outside
    masks[:, :, -1] = False
    masked = ~masks.repeat_interleave(16, 1).repeat_interleave(16, 2)[:, :H, :W]
    for _ in range(3):
        _poison_allocator(
            (C, H, W, channels),
            (C, H, W, 1),
            (C, H, W),
            (C, H, W),
            (C, H, W, 3),
            (C, H, W, 1),
            (C, H, W, 1),
        )
        colors, alphas, normals, distort, median, last_ids, median_ids = _raw_scene(
            s, forward, masks
        )
        assert bool(masked.any()) and bool((~masked).any())
        bg = s["backgrounds"][:, None, None, :].expand_as(colors)
        assert torch.equal(colors[masked], bg[masked])
        for t in (alphas, normals, distort, median):
            assert bool((t[masked] == 0).all())
        assert bool((last_ids[masked] == 0).all()) and bool(
            (median_ids[masked] == 0).all()
        )
    # unmasked pixels: as without masks, and the same in both kernels
    full = _raw_scene(s, "pixel")
    out = _raw_scene(s, forward, masks)
    for a, b in zip(out, full):
        assert torch.equal(a[~masked], b[~masked])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("forward", ["pixel", "subtile"])
def test_2dgs_masked_tiles_background_gradient(forward):
    from gsplat.cuda._wrapper import rasterize_to_pixels_2dgs

    s = _garden(4)
    C, H, W = len(s["offsets"]), s["height"], s["width"]
    torch.manual_seed(3)
    masks = torch.rand(s["offsets"].shape, device=device) > 0.4
    backgrounds = s["backgrounds"].clone().requires_grad_()
    _poison_allocator(
        (C, H, W, 4),
        (C, H, W, 1),
        (C, H, W),
        (C, H, W),
        (C, H, W, 3),
        (C, H, W, 1),
        (C, H, W, 1),
    )
    colors, alphas, *_ = rasterize_to_pixels_2dgs(
        s["means2d"],
        s["rt"],
        s["colors"],
        s["opacities"],
        s["normals"],
        torch.zeros_like(s["means2d"]),
        W,
        H,
        16,
        s["offsets"],
        s["ids"],
        backgrounds=backgrounds,
        masks=masks,
        forward=forward,
    )
    v = torch.randn_like(colors)
    (colors * v).sum().backward()
    expected = (v * (1.0 - alphas.detach())).sum(dim=(1, 2))
    torch.testing.assert_close(backgrounds.grad, expected, rtol=1e-5, atol=1e-5)
    assert bool(torch.isfinite(backgrounds.grad).all())


def _random_tile_surfels(seed, n=2048):
    """Surfels for one 16x16 tile: K [s1 R e_x, s2 R e_y, c] rows as ray
    transforms, with grazing (edge-on) and huge splats, centres near, far and
    behind the camera, means2d on and off the projected centre, opacities just
    above 1/255."""
    g = torch.Generator().manual_seed(seed)
    u = lambda *shape: torch.rand(*shape, generator=g)
    K = torch.tensor([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    w_, x, y, z = q.unbind(-1)
    R = torch.stack(
        (
            torch.stack(
                (1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y)),
                -1,
            ),
            torch.stack(
                (2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x)),
                -1,
            ),
            torch.stack(
                (2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y)),
                -1,
            ),
        ),
        -2,
    )
    kind = torch.randint(0, 4, (n,), generator=g)
    # make a quarter of the splats grazing: normal nearly orthogonal to the view
    graze = kind == 1
    R[graze, :, 1] = R[graze, :, 1] * torch.tensor([1.0, 1.0, 1e-3])
    depth = torch.where(
        kind == 2, (u(n) - 0.3) * 2.0, 0.5 + u(n) * 20.0
    )  # some cross z=0
    centre = torch.stack(
        (
            (u(n) - 0.5) * 2.0 * depth.abs().clamp_min(0.5),
            (u(n) - 0.5) * 2.0 * depth.abs().clamp_min(0.5),
            depth,
        ),
        -1,
    )
    s1 = 0.01 * 10 ** (u(n) * 3)
    s2 = s1 * 10 ** ((u(n) - 0.5) * 2)
    A = K @ torch.stack(
        (s1[:, None] * R[:, :, 0], s2[:, None] * R[:, :, 1], centre), -1
    )
    rt = A.reshape(n, 9)
    proj = A[:, :2, 2] / A[:, 2:, 2].where(A[:, 2:, 2].abs() > 1e-6, torch.ones(1))
    means2d = torch.where(
        (kind == 3)[:, None], (u(n, 2) - 0.5) * 80.0, proj + (u(n, 2) - 0.5) * 4.0
    )
    opac = torch.where(u(n) < 0.6, (1 + (u(n) - 0.2) * 0.05) / 255, u(n) * 0.3)
    colors = u(n, 4)
    normals = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    to = lambda t: t[None].contiguous().to(device)
    return to(means2d), to(rt.reshape(n, 3, 3)), to(colors), to(opac), to(normals)


def _tile_raw(inputs, forward):
    n = inputs[0].shape[1]
    return _raw(
        *inputs,
        16,
        16,
        torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        torch.arange(n, dtype=torch.int32, device=device),
        forward,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("seed", range(12))
def test_2dgs_subtile_random_single_tile(seed):
    inputs = _random_tile_surfels(seed)
    ref = _tile_raw(inputs, "pixel")
    assert float(ref[1].max()) > 0  # the default kernel blends something
    _assert_same(_tile_raw(inputs, "subtile"), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_2dgs_subtile_nonfinite_inputs():
    means2d, rt, colors, opac, normals = _random_tile_surfels(100, n=1024)
    means2d[0, 0::97, 0] = float("nan")
    rt[0, 1::89, 0, 1] = float("inf")
    rt[0, 2::83, 2, 2] = float("nan")
    opac[0, 3::79] = float("nan")
    opac[0, 4::73] = float("inf")
    # a NaN weight passes the per-pixel test whatever the opacity
    means2d[0, 5::71] = float("nan")
    rt[0, 5::71, 0, 0] = float("nan")
    opac[0, 5::71] = 0.0
    inputs = (means2d, rt, colors, opac, normals)
    _assert_same(_tile_raw(inputs, "subtile"), _tile_raw(inputs, "pixel"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_2dgs_unknown_forward_and_tile_size():
    from gsplat.cuda._wrapper import rasterize_to_pixels_2dgs

    s = _garden(4)
    args = (
        s["means2d"],
        s["rt"],
        s["colors"],
        s["opacities"],
        s["normals"],
        torch.zeros_like(s["means2d"]),
        s["width"],
        s["height"],
    )
    with pytest.raises(ValueError, match="forward kernel"):
        rasterize_to_pixels_2dgs(
            *args, 16, s["offsets"], s["ids"], forward="subtile_masks"
        )
    tw, th = math.ceil(s["width"] / 8), math.ceil(s["height"] / 8)
    offsets8 = torch.zeros(
        (len(s["offsets"]), th, tw), dtype=torch.int32, device=device
    )
    with pytest.raises(RuntimeError, match="tile_size 16"):
        rasterize_to_pixels_2dgs(*args, 8, offsets8, s["ids"], forward="subtile")
