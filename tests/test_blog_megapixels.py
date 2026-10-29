"""CPU-only tests for megapixel CLI parsing and camera derivation."""

import importlib.util
import argparse
import json
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/blog_eval/benchmark_megapixels.py"
    )
    spec = importlib.util.spec_from_file_location("megapixels_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_resolution_and_camera_indices(benchmark):
    assert benchmark.parse_resolution("1920x1080") == (1920, 1080)
    assert benchmark.parse_camera_indices(["0,2552", "5103"]) == [0, 2552, 5103]
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_resolution("0x720")


def test_parse_defaults_and_overrides(benchmark, tmp_path):
    args = benchmark.parse_args(
        [
            "--extension",
            "x.so",
            "--ply",
            "a.ply",
            "--cameras",
            "a.json",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert args.camera_indices == [0, 2552, 5103]
    assert args.resolutions == benchmark.DEFAULT_RESOLUTIONS
    assert args.warmup == 20 and args.repeats == 40
    args = benchmark.parse_args(
        [
            "--extension",
            "x.so",
            "--ply",
            "a.ply",
            "--cameras",
            "a.json",
            "--output",
            "o.json",
            "--camera-indices",
            "2",
            "4",
            "--resolutions",
            "3200x1800",
            "--scope",
            "raster",
        ]
    )
    assert args.camera_indices == [2, 4]
    assert args.resolutions == [(3200, 1800)]
    assert args.scope == "raster"


@pytest.mark.parametrize("value", ["0x720", "640x-1", "640", "1x2x3", "axb"])
def test_bad_resolution(benchmark, value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_resolution(value)


@pytest.mark.parametrize(
    "extra",
    [
        ["--repeats", "0"],
        ["--warmup", "-1"],
        ["--camera-indices", "-1"],
        ["--skip-validation", "--save-reference-dir", "/tmp/opencode/refs"],
        ["--skip-validation", "--compare-reference-dir", "/tmp/opencode/refs"],
    ],
)
def test_bad_cli(benchmark, extra):
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--extension",
                "x.so",
                "--ply",
                "s.ply",
                "--cameras",
                "c.json",
                "--output",
                "o.json",
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "shape", [(7,), (7, 2), (7, 4, 3), (1, 7, 2), (1, 7), (1, 7, 4)]
)
def test_lossless_gradient_rows_roundtrip(benchmark, shape, tmp_path):
    value = torch.zeros(shape)
    rows = value.reshape(7, -1)
    rows[1] = torch.arange(rows.shape[1], dtype=torch.float32) + 0.125
    rows[5] = -0.375
    compressed = benchmark.compress_gradient(value, 7)
    assert compressed["shape"] == list(shape)
    assert compressed["indices"].tolist() == [1, 5]
    assert compressed["values"].dtype == torch.float32
    path = tmp_path / "gradient.pt"
    torch.save(compressed, path)
    compressed = torch.load(path, weights_only=True)
    restored = torch.zeros_like(rows)
    restored[compressed["indices"]] = compressed["values"]
    assert torch.equal(restored.reshape(shape), value)
    metric = benchmark.compare_gradient(value, compressed)
    assert metric["finite"] and metric["exact_equal"] and metric["alignment_passed"]
    assert metric["rel_l2"] == 0


def test_new_nonzero_rows_are_not_ignored(benchmark):
    baseline = torch.zeros(6, 2)
    baseline[1] = torch.tensor([3.0, 4.0])
    actual = baseline.clone()
    actual[1, 0] += 1
    actual[5, 1] = 2
    metric = benchmark.compare_gradient(
        actual, benchmark.compress_gradient(baseline, 6)
    )
    assert metric["rel_l2"] == pytest.approx(math.sqrt(5) / 5)
    assert metric["max_abs"] == 2
    assert not metric["exact_equal"] and not metric["alignment_passed"]


def test_zero_reference_and_nonfinite_gradients(benchmark):
    value = torch.zeros(4)
    reference = benchmark.compress_gradient(value, 4)
    assert reference["values"].shape == (0, 1)
    assert benchmark.compare_gradient(value, reference)["exact_equal"]
    value[3] = 1
    assert benchmark.compare_gradient(value, reference)["rel_l2"] == pytest.approx(1e30)
    for bad in (float("nan"), float("inf")):
        value[3] = bad
        result = benchmark.compare_gradient(value, reference)
        assert not result["finite"] and result["rel_l2"] is None
        assert not result["alignment_passed"]
        json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="shape differs"):
        benchmark.compare_gradient(torch.zeros(4, 1), reference)


def test_nonfinite_reference_gradient_rejected(benchmark):
    reference = benchmark.compress_gradient(torch.tensor([float("nan"), 0.0]), 2)
    assert not benchmark.compare_gradient(torch.zeros(2), reference)["finite"]


def test_rgb_psnr_excludes_depth_but_alignment_does_not(benchmark):
    baseline = torch.zeros(1, 2, 3, 4)
    actual = baseline.clone()
    actual[..., 3] = 10
    result = benchmark.compare_image(actual, baseline, rgb_depth=True)
    assert result["mse"] == 0 and result["psnr"] is None
    assert result["depth_max_abs"] == 10 and result["max_abs"] == 10
    assert not result["exact_equal"] and not result["alignment_passed"]
    actual[..., :3] = 0.01
    result = benchmark.compare_image(actual, baseline, rgb_depth=True)
    assert result["psnr"] == pytest.approx(40)
    exact = benchmark.compare_image(baseline, baseline, rgb_depth=True)
    assert exact["exact_equal"] and exact["psnr"] is None
    json.dumps(exact, allow_nan=False)


def test_fixed_gates(benchmark):
    ref = torch.ones(8)
    compressed = benchmark.compress_gradient(ref, 8)
    assert benchmark.compare_gradient(ref + 5e-5, compressed)["alignment_passed"]
    assert not benchmark.compare_gradient(ref + 2e-4, compressed)["alignment_passed"]
    assert benchmark.compare_image(ref + 1e-5, ref)["alignment_passed"]
    assert not benchmark.compare_image(ref + 3e-5, ref)["alignment_passed"]
    assert benchmark.FORWARD_MAX_ABS == 2e-5
    assert benchmark.GRADIENT_REL_L2 == 1e-4


@pytest.mark.parametrize("width,height", [(640, 360), (3200, 1800), (1280, 900)])
def test_fov_and_rotated_camera_cpu_tensors(benchmark, width, height):
    camera = {
        "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "position": [1, 2, 3],
        "fx": 1000,
        "fy": 1100,
        "cx": 930,
        "cy": 500,
        "width": 1920,
        "height": 1080,
    }
    view_np, K_np = benchmark.camera_matrices(camera, width, height)
    view, K = torch.from_numpy(view_np), torch.from_numpy(K_np)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(camera["rotation"])
    c2w[:3, 3] = torch.tensor(camera["position"])
    torch.testing.assert_close(view @ c2w, torch.eye(4))
    assert float(K[0, 0] / width) == pytest.approx(camera["fx"] / camera["width"])
    assert float(K[1, 1] / height) == pytest.approx(camera["fy"] / camera["height"])
    assert float(K[0, 2] / width) == pytest.approx(camera["cx"] / camera["width"])
    assert float(K[1, 2] / height) == pytest.approx(camera["cy"] / camera["height"])


@pytest.mark.parametrize("train", [True, False])
def test_measure_event_order_without_cuda(benchmark, train):
    calls = []

    class Event:
        def __init__(self, index):
            self.index = index

        def record(self):
            calls.append(f"event{self.index}")

        def synchronize(self):
            calls.append(f"sync{self.index}")

        def elapsed_time(self, other):
            return float(other.index - self.index)

    class FakeCuda:
        def synchronize(self):
            calls.append("warmup_sync")

        def reset_peak_memory_stats(self):
            calls.append("reset_peak")

        def max_memory_allocated(self):
            return 1024

    value = torch.tensor([2.0], requires_grad=True)
    value.register_hook(lambda gradient: calls.append("backward") or gradient)

    def render(training):
        assert training == train
        assert torch.is_grad_enabled() == train
        calls.append("forward")
        value.absgrad = torch.ones_like(value)
        return (
            value.square(),
            value * 3,
            {"means2d": value, "flatten_ids": torch.tensor([0])},
        )

    result = benchmark._measure(
        render,
        {"means": value},
        (torch.ones(1), torch.ones(1)),
        0,
        2,
        train,
        FakeCuda(),
        [Event(i) for i in range(3)],
    )
    per_iteration = ["reset_peak", "event0", "forward", "event1"]
    if train:
        per_iteration.append("backward")
    per_iteration += ["event2", "sync2"]
    assert calls == ["warmup_sync", *per_iteration, *per_iteration]
    assert result["forward"]["samples_ms"] == [1, 1]
    assert result["backward"]["samples_ms"] == [1, 1]
    assert result["combined"]["samples_ms"] == [2, 2]
    assert all(result["finite_checks"].values())
    assert value.grad is None and not hasattr(value, "absgrad")


def test_camera_matrices_scales_intrinsics_and_preserves_pose(benchmark):
    camera = {
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "position": [1, 2, 3],
        "fx": 100,
        "fy": 120,
        "cx": 50,
        "cy": 40,
        "width": 100,
        "height": 80,
    }
    view, K = benchmark.camera_matrices(camera, 200, 160)
    assert K.tolist() == [[200.0, 0.0, 100.0], [0.0, 240.0, 80.0], [0.0, 0.0, 1.0]]
    assert view.tolist() == [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -2.0],
        [0.0, 0.0, 1.0, -3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
