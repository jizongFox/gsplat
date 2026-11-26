"""Fused projection backward + sparse Adam (PennyGS, arXiv 2611.27525 Sec. 3.7).

The fused step must equal projection backward through autograd followed by the
SelectiveAdam update on visible Gaussians, leave unseen Gaussians untouched, and
keep bf16 moments close to the float32 ones.
"""

import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")
LRS = (1.6e-4, 1e-3, 5e-3)
BETAS = (0.9, 0.999)
EPS = 1e-15


def _scene():
    means, quats, scales, _, _, viewmats, Ks, width, height = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return means, quats, torch.log(scales * 0.1), viewmats[0], Ks[0], width, height


def _upstream(n, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(shape, device=device, generator=generator)
        for shape in ((n, 2), (n,), (n, 3))
    )


def _reference_step(params, moments, viewmat, K, width, height, upstream):
    """Autograd projection backward + SelectiveAdam (no bias correction)."""
    from gsplat.cuda._wrapper import fully_fused_projection

    leaves = [p.detach().clone().requires_grad_() for p in params]
    means, quats, log_scales = leaves
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, log_scales.exp(), viewmat[None], K[None], width, height
    )
    torch.autograd.backward((means2d[0], depths[0], conics[0]), upstream)
    visible = radii[0] > 0
    out_params, out_moments = [], []
    for k, (p, leaf, lr) in enumerate(zip(params, leaves, LRS)):
        m, v = moments[2 * k].clone(), moments[2 * k + 1].clone()
        g = leaf.grad
        m[visible] = BETAS[0] * m[visible] + (1 - BETAS[0]) * g[visible]
        v[visible] = BETAS[1] * v[visible] + (1 - BETAS[1]) * g[visible] ** 2
        new = p.clone()
        new[visible] -= lr * m[visible] / (v[visible].sqrt() + EPS)
        out_params.append(new)
        out_moments += [m, v]
    return out_params, out_moments, radii[0], conics[0]


def _fused_step(params, moments, viewmat, K, width, height, upstream):
    from gsplat.cuda._wrapper import fully_fused_projection, fused_projection_adam

    with torch.no_grad():
        radii, _, _, conics, _ = fully_fused_projection(
            params[0],
            None,
            params[1],
            params[2].exp(),
            viewmat[None],
            K[None],
            width,
            height,
        )
    params = [p.clone() for p in params]
    moments = [m.clone() for m in moments]
    fused_projection_adam(
        *params,
        viewmat,
        K,
        width,
        height,
        radii[0],
        conics[0],
        *upstream,
        moments=tuple(moments),
        lrs=LRS,
        betas=BETAS,
        eps=EPS,
    )
    return params, moments, radii[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fused_step_matches_autograd_plus_selective_adam():
    means, quats, log_scales, viewmat, K, width, height = _scene()
    params = [means, quats, log_scales]
    moments = [torch.zeros_like(p) for p in params for _ in range(2)]
    for step in range(3):
        upstream = _upstream(len(means), seed=step)
        ref_params, ref_moments, radii, _ = _reference_step(
            params, moments, viewmat, K, width, height, upstream
        )
        params, moments, radii_f = _fused_step(
            params, moments, viewmat, K, width, height, upstream
        )
        assert torch.equal(radii, radii_f)
        # Same math in another evaluation order (and fast-math exp): the mean
        # gradient of tiny Gaussians cancels terms scaled by 1/scale^2, so single
        # elements may differ in the last ~10 bits; gate whole tensors (PR9 style).
        for a, b in zip(params + moments, ref_params + ref_moments):
            rel = float((a - b).double().norm() / b.double().norm().clamp_min(1e-30))
            assert rel < 1e-4, rel  # measured: <= 1.3e-05
    assert 0 < int((radii > 0).sum()) < len(means)  # both cases exercised


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_unseen_gaussians_are_untouched_and_bf16_moments_track_fp32():
    means, quats, log_scales, viewmat, K, width, height = _scene()
    params = [means, quats, log_scales]
    upstream = _upstream(len(means), seed=7)
    fp32 = [torch.full_like(p, 0.5) for p in params for _ in range(2)]
    bf16 = [m.to(torch.bfloat16) for m in fp32]
    params32, moments32, radii = _fused_step(
        params, fp32, viewmat, K, width, height, upstream
    )
    params16, moments16, _ = _fused_step(
        params, bf16, viewmat, K, width, height, upstream
    )
    unseen = radii <= 0
    for new, old in zip(params32 + params16, params + params):
        assert torch.equal(new[unseen], old[unseen])
    for m in moments16:
        assert m.dtype == torch.bfloat16
        assert bool((m[unseen] == 0.5).all())
    for m16, m32 in zip(moments16, moments32):
        torch.testing.assert_close(m16.float(), m32, rtol=1e-2, atol=1e-6)
    for p16, p32, p in zip(params16, params32, params):
        step32 = (p32 - p)[~unseen]
        step16 = (p16 - p)[~unseen]
        assert float((step16 - step32).norm() / step32.norm()) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fused_step_rejects_mixed_moment_types():
    from gsplat.cuda._wrapper import fused_projection_adam

    means, quats, log_scales, viewmat, K, width, height = _scene()
    params = [means, quats, log_scales]
    moments = [torch.zeros_like(p) for p in params for _ in range(2)]
    moments[3] = moments[3].to(torch.bfloat16)
    n = len(means)
    with pytest.raises(AssertionError, match="mixed"):
        fused_projection_adam(
            *params,
            viewmat,
            K,
            width,
            height,
            torch.zeros(n, dtype=torch.int32, device=device),
            torch.zeros(n, 3, device=device),
            *_upstream(n, 0),
            moments=tuple(moments),
            lrs=LRS,
        )
