"""CPU-only controls for the standalone seven-channel benchmark."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks/warp_private/benchmark_seven_channel.py"
    spec = importlib.util.spec_from_file_location("seven_channel_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_and_resolution_controls(benchmark):
    args = benchmark.parse_args(["--extension", "base.so", "--output", "result.json"])
    assert args.camera_indices == [0, 623, 1247]
    assert args.resolutions == ["native"]
    assert args.channels == 7 and args.tile_size == 16
    assert args.warmup == 20 and args.repeats == 40
    assert str(args.ply).endswith("56ee2948/point_cloud/iteration_020000/point_cloud.ply")
    args = benchmark.parse_args(["--extension", "x.so", "--output", "o.json", "--channels", "4",
                                 "--resolutions", "native", "4k", "640x360", "--camera-indices", "2,4", "8"])
    assert args.resolutions == ["native", "4k", (640, 360)]
    assert args.camera_indices == [2, 4, 8]


@pytest.mark.parametrize("extra", [
    ["--save-reference-dir", "/tmp/opencode/base", "--compare-reference-dir", "/tmp/opencode/base"],
    ["--save-reference-dir", str(Path(__file__).resolve().parents[1] / "refs")],
    ["--warmup", "-1"], ["--repeats", "0"],
    ["--tile-size", "8"], ["--camera-indices", "-1"], ["--resolutions", "0x720"],
])
def test_invalid_cli(benchmark, extra):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--extension", "x.so", "--output", "o.json", *extra])


def write_colmap(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 1280 720 500 510 640 360\n2 SIMPLE_PINHOLE 100 80 50 45 35\n")
    (tmp_path / "images.txt").write_text(
        "# poses\n9 1 0 0 0 1 2 3 1 z.jpg\n\n"
        "4 0.7071067811865476 0 0 0.7071067811865476 2 3 4 2 a.jpg\n10 20 -1\n")


def test_colmap_conversion_sort_pose_and_intrinsics(benchmark, tmp_path):
    write_colmap(tmp_path)
    cameras = benchmark.colmap_cameras(tmp_path)
    assert [c["id"] for c in cameras] == [4, 9]
    assert cameras[0]["fx"] == cameras[0]["fy"] == 50
    view, K = benchmark.pr9.camera_matrices(cameras[0], 200, 160)
    np.testing.assert_allclose(view[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(view[:3, 3], [2, 3, 4])
    assert K.tolist() == [[100, 0, 90], [0, 100, 70], [0, 0, 1]]
    assert cameras[1]["position"] == [-1, -2, -3]
    assert cameras[1]["width"] == 1280 and cameras[1]["height"] == 720


def test_colmap_rejects_distortion_and_missing_observation(benchmark, tmp_path):
    write_colmap(tmp_path)
    path = tmp_path / "cameras.txt"
    path.write_text(path.read_text().replace("PINHOLE 1280", "OPENCV 1280"))
    with pytest.raises(ValueError, match="Unsupported camera model"):
        benchmark.colmap_cameras(tmp_path)
    write_colmap(tmp_path)
    (tmp_path / "images.txt").write_text("9 1 0 0 0 1 2 3 1 z.jpg\n")
    with pytest.raises(ValueError, match="observation"):
        benchmark.colmap_cameras(tmp_path)


def test_local_z_normal_uses_wxyz_camera_rotation_and_no_flip(benchmark):
    q = torch.tensor([[2., 0, 0, 0], [math.sqrt(.5), 0, math.sqrt(.5), 0], [0., 1, 0, 0]])
    view = torch.eye(4)[None]
    view[0, :3, :3] = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    view[0, :3, 3] = 500  # translation must not affect normals
    normal = benchmark.camera_normals(q, view)
    torch.testing.assert_close(normal, torch.tensor([[[0., 0, 1], [0., 1, 0], [0., 0, -1]]]), atol=3e-7, rtol=1e-6)
    for bad in (torch.zeros(1, 4), torch.full((1, 4), float("nan"))):
        with pytest.raises(ValueError, match="quaternion"):
            benchmark.camera_normals(bad, view)


@pytest.mark.parametrize("channels", [3, 4, 7, 8])
def test_real_attributes_and_wrapper_owns_padding(benchmark, channels):
    class Wrapper:
        @staticmethod
        def spherical_harmonics(degree, dirs, coefficients, masks):
            return torch.tensor([[[-1., -.5, .25]]])

    raw = {
        "means": torch.tensor([[0.0, 0.0, 1.0]]),
        "quats": torch.tensor([[1.0, 0, 0, 0]]),
        "log_scales": torch.zeros(1, 3),
        "sh_coefficients": torch.zeros(1, 1, 3),
    }
    meta = {"radii": torch.ones(1, 1), "depths": torch.tensor([[5.]])}
    result = benchmark.attributes(raw, torch.eye(4)[None], meta, 0, Wrapper, channels)
    assert result.shape == (1, 1, channels)  # no harness-side synthetic eighth attribute
    expected = torch.tensor([[[-.5, 0, .75, 5, 0, 0, 0, 0]]])
    torch.testing.assert_close(result, expected[..., :channels])


def test_image_gates_include_normal_and_depth_but_psnr_is_rgb(benchmark):
    baseline = torch.zeros(1, 2, 3, 7)
    actual = baseline.clone()
    actual[..., 4] = 3e-5
    metric = benchmark.image_metrics(actual, baseline)
    assert not metric["alignment_passed"]
    assert metric["depth_max_abs"] == 0
    assert metric["normal_max_abs"] == pytest.approx(3e-5)
    assert metric["mse"] == 0 and metric["psnr"] is None
    actual.zero_()[..., 3] = 3e-5
    assert not benchmark.image_metrics(actual, baseline)["alignment_passed"]
    actual.zero_()[..., :3] = .01
    assert benchmark.image_metrics(actual, baseline)["psnr"] == pytest.approx(40)
    actual[..., 6] = float("nan")
    metric = benchmark.image_metrics(actual, baseline)
    assert not metric["finite"] and not metric["alignment_passed"]
    json.dumps(metric, allow_nan=False)
    assert benchmark.pr9.FORWARD_MAX_ABS == 2e-5
    assert benchmark.pr9.GRADIENT_REL_L2 == 1e-4


def test_gradient_all_rows_and_identity_control(benchmark):
    baseline = torch.zeros(1, 10001, 7)
    baseline[0, 1, 0] = 1
    reference = benchmark.pr9.compress_gradient(baseline, 10001)
    assert benchmark.pr9.compare_gradient(baseline, reference)["exact_equal"]
    actual = baseline.clone()
    actual[0, -1, -1] = 2e-4  # previously zero row, far beyond a prefix sample
    assert not benchmark.pr9.compare_gradient(actual, reference)["alignment_passed"]


def test_reference_identity_rejects_content_and_settings_changes(benchmark):
    signature = {"ply_sha256": "a", "colmap_sha256": {"images.txt": "b"}, "seed": 9,
                 "camera": {"id": 1}, "resolution": [1280, 720], "channels": 7,
                 "normal_convention": benchmark.NORMAL_CONVENTION, "settings": {"tile_size": 16}}
    reference = {"signature": copy.deepcopy(signature), "repeat_noise": {"image": {"alignment_passed": True}}}
    benchmark.validate_reference(reference, signature)
    for key in signature:
        changed = copy.deepcopy(signature)
        changed[key] = "different"
        with pytest.raises(ValueError, match="contents or settings"):
            benchmark.validate_reference(reference, changed)
    reference["repeat_noise"]["image"]["alignment_passed"] = False
    with pytest.raises(ValueError, match="repeat correctness"):
        benchmark.validate_reference(reference, signature)


def test_seeded_upstream_identity_independent_of_case_order(benchmark):
    seed, first = benchmark.upstream_values(2026, 623, 8, 4, 7, "cpu")
    benchmark.upstream_values(2026, 0, 8, 4, 7, "cpu")
    same_seed, second = benchmark.upstream_values(2026, 623, 8, 4, 7, "cpu")
    assert same_seed == seed
    assert first[0].shape == (1, 4, 8, 7) and first[1].shape == (1, 4, 8, 1)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    _, other = benchmark.upstream_values(2026, 0, 8, 4, 7, "cpu")
    assert not torch.equal(first[0], other[0])


def test_raster_worker_reference_roundtrip_and_failure_without_cuda(benchmark, tmp_path, monkeypatch):
    """Exercise the real event/snapshot helpers using a differentiable CPU raster."""
    calls = []
    drift = [0.0]

    class Event:
        def record(self):
            calls.append("event")

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    class Cuda:
        def synchronize(self):
            pass

        def reset_peak_memory_stats(self):
            pass

        def max_memory_allocated(self):
            return 0

    class Wrapper:
        @staticmethod
        def spherical_harmonics(*args, **kwargs):
            return torch.zeros(1, 2, 3)

        @staticmethod
        def rasterize_to_pixels(means, conics, colors, opacities, width, height, tile_size,
                                offsets, ids, packed, absgrad):
            calls.append("raster")
            assert not packed and absgrad and colors.shape[-1] == 7
            if torch.is_grad_enabled():
                def save_absgrad(gradient):
                    means.absgrad = gradient.abs()
                means.register_hook(save_absgrad)
            scalar = means.sum() + conics.sum() + opacities.sum()
            image = (colors.sum(dim=1) + scalar).reshape(1, 1, 1, 7).expand(1, height, width, 7)
            return image + drift[0], scalar.expand(1, height, width, 1)

    def prepare(*args, **kwargs):
        calls.append("prepare")
        assert not torch.is_grad_enabled()
        return None, None, {"means2d": torch.zeros(1, 2, 2), "conics": torch.ones(1, 2, 3),
                            "opacities": torch.ones(1, 2), "radii": torch.ones(1, 2),
                            "depths": torch.ones(1, 2), "flatten_ids": torch.tensor([0, 1]),
                            "isect_offsets": torch.zeros(1, 1, 1, dtype=torch.int32)}

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Event())
    measure = benchmark.pr9._measure
    monkeypatch.setattr(benchmark.pr9, "_measure",
                        lambda run, inputs, upstream, warmup, repeats, train, cuda, events:
                        measure(run, inputs, upstream, warmup, repeats, train, Cuda(), events))
    upstream_values = benchmark.upstream_values
    monkeypatch.setattr(benchmark, "upstream_values",
                        lambda seed, index, width, height, channels, device:
                        upstream_values(seed, index, width, height, channels, "cpu"))
    args = benchmark.parse_args(["--extension", "base.so", "--output", "o.json", "--warmup", "0",
                                 "--repeats", "2", "--save-reference-dir", str(tmp_path / "refs")])
    args.provenance = {"extension_sha256": "baseline"}
    raw = dict(means=torch.zeros(2, 3), quats=torch.tensor([[1., 0, 0, 0]]).expand(2, 4),
               log_scales=torch.zeros(2, 3), opacity_logits=torch.zeros(2), sh_coefficients=torch.zeros(2, 1, 3))
    camera = dict(width=8, height=4, fx=5, fy=5, cx=4, cy=2, rotation=np.eye(3).tolist(), position=[0, 0, 0])
    settings = dict(packed=False, absgrad=True, tile_size=16, compact_box=True, compact_box_impl="sweep")

    def case():
        return benchmark.run_case(args, raw, camera, 0, (8, 4), 0, settings, {"ply_sha256": "input"}, prepare, Wrapper)

    baseline = case()
    assert baseline["correctness_passed"] and baseline["alignment_passed"] is None
    assert calls[0] == "prepare" and calls.count("prepare") == 1
    assert calls.index("prepare") < calls.index("event") < calls.index("raster")
    assert set(baseline["timing"]) == {"forward_only", "training_forward", "backward", "combined"}
    assert all(len(m["samples_ms"]) == 2 for m in baseline["timing"].values())
    args.compare_reference_dir = args.save_reference_dir
    args.save_reference_dir = None
    identity = case()
    assert identity["alignment_passed"]
    assert all(m["exact_equal"] for m in identity["parity"].values())
    assert set(identity["parity"]) == {"image", "alpha", "grad.means2d", "grad.conics", "grad.colors",
                                       "grad.opacities", "grad.means2d_abs"}
    drift[0] = 1e-3
    changed = case()
    assert changed["status"] == "correctness_failed" and not changed["alignment_passed"]
