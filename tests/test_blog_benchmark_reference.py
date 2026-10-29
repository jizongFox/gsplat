"""CPU-only checks for the blog benchmark's reference protocol."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/blog_eval/gsplat_blog_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("blog_benchmark_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(benchmark, tmp_path):
    ply = tmp_path / "scene.ply"
    cameras = tmp_path / "cameras.json"
    # Signature tests hash bytes; no PLY parsing, CUDA or scene loading is needed.
    ply.write_bytes(b"PLY input A")
    cameras.write_bytes(b"camera input A")
    return benchmark.parse_args(["--ply", str(ply), "--cameras", str(cameras)])


@pytest.mark.parametrize("same_path", [True, False])
def test_reference_actions_are_exclusive_before_loading(
    benchmark, args, tmp_path, monkeypatch, capsys, same_path
):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"existing baseline evidence")
    output = baseline if same_path else tmp_path / "new-reference.pt"

    def unexpected(*_args, **_kwargs):
        pytest.fail("Conflicting reference actions must fail before loading")

    monkeypatch.setattr(benchmark, "load_scene", unexpected)
    monkeypatch.setattr(benchmark.importlib.util, "spec_from_file_location", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--ply",
            str(args.ply),
            "--cameras",
            str(args.cameras),
            "--extension",
            str(tmp_path / "unused.so"),
            "--save-reference",
            str(output),
            "--compare-reference",
            str(baseline),
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert baseline.read_bytes() == b"existing baseline evidence"
    if not same_path:
        assert not output.exists()


@pytest.mark.parametrize("action", [None, "--save-reference", "--compare-reference"])
def test_individual_reference_actions_remain_valid(benchmark, args, tmp_path, action):
    argv = ["--ply", str(args.ply), "--cameras", str(args.cameras)]
    if action:
        argv.extend([action, str(tmp_path / "reference.pt")])
    parsed = benchmark.parse_args(argv)
    assert bool(parsed.save_reference) == (action == "--save-reference")
    assert bool(parsed.compare_reference) == (action == "--compare-reference")


def test_unchanged_signature_matches_and_hashes_inputs(benchmark, args):
    saved = benchmark.reference_signature(args, 12, 1280, 720, 1)
    current = benchmark.reference_signature(args, 12, 1280, 720, 1)
    assert saved["protocol_version"] == benchmark.REFERENCE_PROTOCOL_VERSION
    assert saved["ply_sha256"] == hashlib.sha256(args.ply.read_bytes()).hexdigest()
    assert (
        saved["cameras_sha256"] == hashlib.sha256(args.cameras.read_bytes()).hexdigest()
    )
    benchmark.validate_reference_signature({"signature": saved}, current)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("tile_size", 8),
        ("compact_box", True),
        ("compact_box_impl", "rect_min"),
        ("packed", True),
        ("raster_only", True),
        ("render_mode", "RGB+ED"),
        ("absgrad", True),
        ("camera_index", 1),
        ("seed", 42),
    ],
)
def test_changed_rendering_settings_reject_reference(benchmark, args, setting, value):
    saved = benchmark.reference_signature(args, 12, 1280, 720, 1)
    setattr(args, setting, value)
    changed = benchmark.reference_signature(args, 12, 1280, 720, 1)
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, changed)


@pytest.mark.parametrize("input_name", ["ply", "cameras"])
def test_changed_contents_at_same_path_reject_reference(benchmark, args, input_name):
    saved = benchmark.reference_signature(args, 12, 1280, 720, 1)
    path = getattr(args, input_name)
    # Keep pathname and file size unchanged, just as with replacing a checkpoint.
    original = path.read_bytes()
    path.write_bytes(original[:-1] + b"B")
    changed = benchmark.reference_signature(args, 12, 1280, 720, 1)
    assert saved[input_name] == changed[input_name]
    assert len(original) == path.stat().st_size
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, changed)


@pytest.mark.parametrize("version", [None, 1, 3])
def test_incomplete_or_incompatible_protocol_requires_new_reference(
    benchmark, args, version
):
    current = benchmark.reference_signature(args, 12, 1280, 720, 1)
    saved = dict(current)
    if version is None:
        saved.pop("protocol_version")
    else:
        saved["protocol_version"] = version
    with pytest.raises(ValueError, match="protocol version differs; regenerate"):
        benchmark.validate_reference_signature({"signature": saved}, current)


def test_missing_workload_setting_is_not_silently_accepted(benchmark, args):
    current = benchmark.reference_signature(args, 12, 1280, 720, 1)
    saved = dict(current)
    saved.pop("compact_box")
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, current)
