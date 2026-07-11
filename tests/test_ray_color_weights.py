import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


def test_ray_color_weights_add_detached_per_channel_gradient():
    from gsplat.cuda._wrapper import rasterize_to_pixels

    device = torch.device("cuda:0")
    image_width = image_height = tile_size = 16
    means2d = torch.tensor(
        [[[8.0, 8.0], [8.0, 8.0]]], device=device, requires_grad=True
    )
    conics = torch.tensor(
        [[[0.1, 0.0, 0.1], [0.1, 0.0, 0.1]]],
        device=device,
        requires_grad=True,
    )
    colors = torch.tensor(
        [[[0.2, 0.6, 0.9], [0.8, 0.1, 0.4]]],
        device=device,
        requires_grad=True,
    )
    opacities = torch.tensor([[0.4, 0.3]], device=device, requires_grad=True)
    background = torch.tensor([[0.1, 0.3, 0.7]], device=device)
    isect_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    channel_weights = torch.tensor([0.25, 0.0, 0.5], device=device)

    rendered, _ = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=background,
        ray_color_weights=channel_weights,
    )
    (rendered.sum() * 0.0).backward()

    ys, xs = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(image_width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    sigma = 0.05 * ((8.0 - xs).square() + (8.0 - ys).square())
    alpha1 = 0.4 * torch.exp(-sigma)
    alpha2 = 0.3 * torch.exp(-sigma)
    alpha1 = torch.where(alpha1 >= 1.0 / 255.0, alpha1, 0.0)
    alpha2 = torch.where(alpha2 >= 1.0 / 255.0, alpha2, 0.0)
    weight1 = alpha1
    weight2 = (1.0 - alpha1) * alpha2
    final_transmittance = (1.0 - alpha1) * (1.0 - alpha2)
    final_color = (
        weight1[..., None] * colors.detach()[0, 0]
        + weight2[..., None] * colors.detach()[0, 1]
        + final_transmittance[..., None] * background[0]
    )
    expected1 = (
        weight1[..., None] * channel_weights * (colors.detach()[0, 0] - final_color)
    ).sum(dim=(0, 1))
    expected2 = (
        weight2[..., None] * channel_weights * (colors.detach()[0, 1] - final_color)
    ).sum(dim=(0, 1))

    torch.testing.assert_close(colors.grad[0, 0], expected1, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(colors.grad[0, 1], expected2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(opacities.grad, torch.zeros_like(opacities.grad))
    torch.testing.assert_close(means2d.grad, torch.zeros_like(means2d.grad))
    torch.testing.assert_close(conics.grad, torch.zeros_like(conics.grad))


def test_ray_color_weights_none_matches_default():
    from gsplat.cuda._wrapper import rasterize_to_pixels

    device = torch.device("cuda:0")
    inputs = {
        "means2d": torch.tensor([[[8.0, 8.0]]], device=device),
        "conics": torch.tensor([[[0.1, 0.0, 0.1]]], device=device),
        "colors": torch.tensor([[[0.2, 0.6, 0.9]]], device=device),
        "opacities": torch.tensor([[0.5]], device=device),
        "image_width": 16,
        "image_height": 16,
        "tile_size": 16,
        "isect_offsets": torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        "flatten_ids": torch.zeros((1,), dtype=torch.int32, device=device),
    }

    default_colors = inputs["colors"].clone().requires_grad_()
    default_render, _ = rasterize_to_pixels(**{**inputs, "colors": default_colors})
    default_render.sum().backward()

    none_colors = inputs["colors"].clone().requires_grad_()
    none_render, _ = rasterize_to_pixels(
        **{**inputs, "colors": none_colors}, ray_color_weights=None
    )
    none_render.sum().backward()

    torch.testing.assert_close(none_render, default_render)
    torch.testing.assert_close(none_colors.grad, default_colors.grad)


@pytest.mark.parametrize("packed", [False, True])
def test_ray_color_weights_keep_cameras_separate(packed: bool):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    device = torch.device("cuda:0")
    means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0]], device=device)
    conics = torch.tensor([[0.1, 0.0, 0.1], [0.1, 0.0, 0.1]], device=device)
    colors = torch.tensor(
        [[0.2, 0.6, 0.9], [0.8, 0.1, 0.4]], device=device, requires_grad=True
    )
    opacities = torch.tensor([0.4, 0.3], device=device)
    if not packed:
        means2d = means2d[:, None]
        conics = conics[:, None]
        colors = colors.detach()[:, None].requires_grad_()
        opacities = opacities[:, None]

    backgrounds = torch.tensor([[0.1, 0.3, 0.7], [0.9, 0.2, 0.0]], device=device)
    channel_weights = torch.tensor([0.25, 0.0, 0.5], device=device)
    rendered, _ = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        16,
        16,
        16,
        torch.tensor([[[0]], [[1]]], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        backgrounds=backgrounds,
        packed=packed,
        ray_color_weights=channel_weights,
    )
    (rendered.sum() * 0.0).backward()

    color_values = colors.detach() if packed else colors.detach()[:, 0]
    ys, xs = torch.meshgrid(
        torch.arange(16, device=device, dtype=torch.float32) + 0.5,
        torch.arange(16, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    visibility = torch.exp(-0.05 * ((8.0 - xs).square() + (8.0 - ys).square()))
    for camera_id in range(2):
        alpha = opacities.reshape(-1)[camera_id] * visibility
        alpha = torch.where(alpha >= 1.0 / 255.0, alpha, 0.0)
        final = (
            alpha[..., None] * color_values[camera_id]
            + (1.0 - alpha[..., None]) * backgrounds[camera_id]
        )
        expected = (
            alpha[..., None] * channel_weights * (color_values[camera_id] - final)
        ).sum(dim=(0, 1))
        actual = colors.grad[camera_id] if packed else colors.grad[camera_id, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_ray_color_weights_use_raw_depth_before_expected_depth_normalization():
    from gsplat.rendering import rasterization

    device = torch.device("cuda:0")
    gradients = []
    for render_mode in ["RGB+D", "RGB+ED"]:
        means = torch.tensor(
            [[0.0, 0.0, 2.0]], device=device, requires_grad=True
        )
        rendered, _, _ = rasterization(
            means=means,
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            scales=torch.full((1, 3), 0.5, device=device),
            opacities=torch.full((1,), 0.5, device=device),
            colors=torch.ones((1, 3), device=device),
            viewmats=torch.eye(4, device=device)[None],
            Ks=torch.tensor(
                [[[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]]],
                device=device,
            ),
            width=16,
            height=16,
            render_mode=render_mode,
            ray_color_weights=torch.tensor([0.0, 0.0, 0.0, 0.5], device=device),
        )
        (rendered.sum() * 0.0).backward()
        gradients.append(means.grad.detach().clone())

    assert gradients[0].abs().sum() > 0
    torch.testing.assert_close(gradients[1], gradients[0])
