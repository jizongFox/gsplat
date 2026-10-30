"""CPU-only renderer parity/protocol checks; never import gsplat or initialize CUDA."""

import ast
import json
from pathlib import Path

import pytest
import torch

import renderer_benchmark as bench
from convert_colmap import convert

PARENT = Path("/home/jizong/workspace/dConstruct/gaussian-splatting/dc_reality")


def source_functions(path, names, namespace):
    """Execute only source formula ASTs, stripping imports, annotations and compile."""
    if not path.exists():
        pytest.skip(f"Read-only renderer source unavailable: {path}")
    tree = ast.parse(path.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                arg.annotation = None
            functions.append(node)
    assert {node.name for node in functions} == set(names)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("seed", [1, 9, 2026])
def test_exact_parent_normal_formula_and_gradients(seed):
    ns = source_functions(PARENT / "geometry/torch/quaternion.py", ["quaternion_normalize"],
                          {"torch": torch, "_check_last_dim": lambda *args: None})
    source_functions(PARENT / "splatting/utils/projection.py",
                     ["_selected_gaussian_axis", "_gaussian_frame_normals_from_tensors", "_align_normals_orientation"], ns)
    generator = torch.Generator().manual_seed(seed)
    quats = torch.randn((90, 4), dtype=torch.float64, generator=generator, requires_grad=True)
    scales = torch.rand((90, 3), dtype=torch.float64, generator=generator) + 0.01
    means = torch.randn((90, 3), dtype=torch.float64, generator=generator, requires_grad=True)
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = bench.quaternion_columns(torch.randn((1, 4), dtype=torch.float64, generator=generator))[0].T
    viewmat[:3, 3] = torch.tensor([0.5, -0.7, 3.0])
    expected, means_cam = ns["_align_normals_orientation"](
        ns["_gaussian_frame_normals_from_tensors"](quats, scales, viewmat), means, viewmat)
    actual = bench.camera_normals(quats, scales, means, viewmat)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(scales.argmin(-1).tolist()) == {0, 1, 2}
    assert bool(((actual * means_cam).sum(-1) >= 0).all())
    upstream = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
    actual_grad = torch.autograd.grad(actual, (quats, means), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (quats, means), upstream)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_minor_axis_ties_post_division_offset_and_zero_sign():
    quats = torch.tensor([[1., 0., 0., 0.]]).expand(4, -1)
    scales = torch.tensor([[1., 2., 3.], [3., 1., 2.], [3., 2., 1.], [1., 1., 1.]])
    base = torch.eye(3)[torch.tensor([0, 1, 2, 0])] + 1e-5
    means = torch.stack((base[0], -base[1], base[2], torch.zeros(3)))
    actual = bench.camera_normals(quats, scales, means, torch.eye(4))
    expected = base * torch.tensor([1., -1., 1., 0.])[:, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Regression against treating every Gaussian as its quaternion's local +Z.
    assert not torch.allclose(actual[:2], torch.tensor([[0., 0., 1.], [0., 0., 1.]]))


def test_full_renderer_calls_six_colors_and_extracts_depth_channel_seven():
    inputs = {
        "means": torch.tensor([[0., 0., 3.]], requires_grad=True),
        "quats": torch.tensor([[1., 0., 0., 0.]], requires_grad=True),
        "log_scales": torch.tensor([[0., 1., 2.]], requires_grad=True),
        "opacity_logits": torch.tensor([0.], requires_grad=True),
        "dc": torch.tensor([[0., 0.1, -0.1]], requires_grad=True),
    }
    seen = {}

    def rasterization(**kwargs):
        seen.update(kwargs)
        assert kwargs["colors"].shape == (1, 6)
        assert kwargs["backgrounds"].shape == (1, 6)
        colors_depth = torch.cat((kwargs["colors"], kwargs["means"][:, 2:3]), -1)
        rendered = colors_depth[None, None].expand(1, 3, 4, 7)
        alpha = kwargs["opacities"].reshape(1, 1, 1, 1).expand(1, 3, 4, 1)
        return rendered, alpha, {"means2d": kwargs["means"][:, :2]}

    K = torch.tensor([[4., 0., 2.], [0., 4., 1.5], [0., 0., 1.]])
    image, alpha, info, normals, surface = bench.render(rasterization, inputs, torch.eye(4), K, 4, 3, True)
    assert image.shape == (1, 3, 4, 4)
    assert alpha.shape == (1, 3, 4, 1)
    torch.testing.assert_close(image[..., 3], torch.full((1, 3, 4), 3.))
    torch.testing.assert_close(seen["colors"][:, :3], inputs["dc"] * 0.28209479177387814 + 0.5)
    torch.testing.assert_close(seen["colors"][:, 3:], bench.camera_normals(inputs["quats"], inputs["log_scales"].exp(), inputs["means"], torch.eye(4)))
    for name, expected in {"packed": False, "sh_degree": None, "render_mode": "RGB+ED", "absgrad": True,
                           "compact_box": True, "compact_box_impl": "sweep", "near_plane": .01, "far_plane": 1000.}.items():
        assert seen[name] == expected
    assert normals.shape == surface.shape == (3, 3, 4)
    assert not torch.cuda.is_initialized()


def test_surface_normal_source_formula():
    ns = source_functions(PARENT / "splatting/utils/projection.py", ["grid_points_to_normal"], {"torch": torch})
    depth = torch.full((1, 5, 6), 3.)
    K = torch.tensor([[5., 0., 3.], [0., 5., 2.5], [0., 0., 1.]])
    y, x = torch.meshgrid(torch.arange(5) + .5, torch.arange(6) + .5, indexing="ij")
    points = torch.einsum("ij,hwj->hwi", K.inverse(), torch.stack((x, y, torch.ones_like(x)), -1)) * depth.permute(1, 2, 0)
    expected = -ns["grid_points_to_normal"](points).permute(2, 0, 1)
    torch.testing.assert_close(bench.surface_normals(depth, K), expected, atol=0, rtol=0)


def test_colmap_world_frame_converter(tmp_path):
    cameras, images = tmp_path / "cameras.txt", tmp_path / "images.txt"
    cameras.write_text("# cameras\n2 PINHOLE 6 5 5 5 3 2.5\n")
    images.write_text("# images\n7 0.7071067811865476 0 0 0.7071067811865476 1 2 3 2 image.png\n\n")
    converted = convert(cameras, images)
    assert converted[0]["id"] == 7
    viewmat, K = bench.camera_tensors(converted[0])
    expected = torch.tensor([[0., -1., 0., 1.], [1., 0., 0., 2.], [0., 0., 1., 3.], [0., 0., 0., 1.]])
    torch.testing.assert_close(viewmat, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(K, torch.tensor([[5., 0., 3.], [0., 5., 2.5], [0., 0., 1.]]))
    cameras.write_text("2 OPENCV 6 5 5 5 3 2.5 0 0 0 0\n")
    with pytest.raises(ValueError, match="undistorted"):
        convert(cameras, images)


def arguments(tmp_path):
    ply, cameras = tmp_path / "scene.ply", tmp_path / "cameras.json"
    ply.write_bytes(b"scene content")
    cameras.write_text(json.dumps([]))
    return ["--ply", str(ply), "--cameras", str(cameras), "--repo", str(tmp_path / "repo"),
            "--extension", str(tmp_path / "csrc.so"), "--output", str(tmp_path / "run.json")]


def test_reference_mutual_exclusion_and_read_only_checkout(tmp_path):
    args = arguments(tmp_path)
    with pytest.raises(SystemExit):
        bench.parse_args(args + ["--save-reference", str(tmp_path / "ref.pt"), "--compare-reference", "other.pt"])
    with pytest.raises(SystemExit):
        bench.parse_args(args + ["--save-reference", str(tmp_path / "repo/ref.pt")])
    ref = tmp_path / "ref.pt"
    ref.touch()
    with pytest.raises(SystemExit):
        bench.parse_args(args + ["--save-reference", str(ref)])


def test_content_hash_settings_and_reference_tensor_integrity(tmp_path):
    args = bench.parse_args(arguments(tmp_path) + ["--save-reference", str(tmp_path / "ref.pt")])
    sig = bench.signature(args)
    tensor = torch.tensor([1.])
    reference = {"signature": sig, "passed": True, "cases": {"case": {"tensors": {"image": tensor}, "tensor_sha256": {"image": bench.tensor_hash(tensor)}}}}
    bench.validate_reference(reference, sig)
    args.ply.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="content hashes"):
        bench.validate_reference(reference, bench.signature(args))
    reference["cases"]["case"]["tensors"]["image"] = tensor + 1
    with pytest.raises(ValueError, match="tensor content hashes"):
        bench.validate_reference(reference, sig)


def test_pr9_limits_apply_to_every_gradient_and_forward_tensor():
    expected = {"image": torch.tensor([1.]), "grad.means": torch.tensor([1.]), "grad.means2d_abs": torch.tensor([0.])}
    actual = {name: value.clone() for name, value in expected.items()}
    assert all(x["passed"] for x in bench.compare(actual, expected).values())
    actual["image"] += 3e-5
    actual["grad.means"] += 2e-4
    actual["grad.means2d_abs"] += 1e-20
    assert not any(x["passed"] for x in bench.compare(actual, expected).values())
    actual["image"][:] = float("nan")
    assert not bench.compare(actual, expected)["image"]["finite"]
    with pytest.raises(ValueError, match="names"):
        bench.compare({"image": actual["image"]}, expected)
