"""CPU-only checks for the blog benchmark's reference protocol."""

import hashlib
import importlib.util
import json
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
    write_scene_files(ply, cameras)
    return benchmark.parse_args(["--ply", str(ply), "--cameras", str(cameras)])


def write_scene_files(ply: Path, cameras: Path, rest_names=None, omit=()):
    if rest_names is None:
        rest_names = [f"f_rest_{index}" for index in range(9)]
    properties = ["x", "y", "z"]
    properties += [f"scale_{i}" for i in range(3)]
    properties += [f"rot_{i}" for i in range(4)]
    properties += ["opacity"]
    properties += [f"f_dc_{i}" for i in range(3)]
    properties += list(rest_names)
    properties = [name for name in properties if name not in set(omit)]
    defaults = {name: 0.0 for name in properties}
    defaults.update(
        {"rot_0": 1.0, "z": 3.0, "f_dc_0": 0.1, "f_dc_1": 0.2, "f_dc_2": 0.3}
    )
    rows = []
    for index in range(12):
        values = dict(defaults)
        values["x"] = index * 0.01
        rows.append(" ".join(str(values[name]) for name in properties))
    header = ["ply", "format ascii 1.0", "element vertex 12"]
    header += [f"property float {name}" for name in properties]
    header += ["end_header"]
    ply.write_text("\n".join(header + rows) + "\n")
    cameras.write_text(
        json.dumps(
            [
                {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "position": [0, 0, 0],
                    "fx": 10,
                    "fy": 10,
                    "cx": 8,
                    "cy": 6,
                    "width": 16,
                    "height": 12,
                }
            ]
        )
    )


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
    saved = benchmark.reference_signature(args, 12, 16, 12, 1)
    current = benchmark.reference_signature(args, 12, 16, 12, 1)
    assert saved["protocol_version"] == benchmark.REFERENCE_PROTOCOL_VERSION
    assert saved["protocol_version"] == 3
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
        ("max_points", 100),
        ("seed", 42),
    ],
)
def test_changed_rendering_settings_reject_reference(benchmark, args, setting, value):
    saved = benchmark.reference_signature(args, 12, 16, 12, 1)
    setattr(args, setting, value)
    changed = benchmark.reference_signature(args, 12, 16, 12, 1)
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, changed)


@pytest.mark.parametrize("input_name", ["ply", "cameras"])
def test_changed_contents_at_same_path_reject_reference(benchmark, args, input_name):
    saved = benchmark.reference_signature(args, 12, 16, 12, 1)
    path = getattr(args, input_name)
    # Keep pathname and file size unchanged, just as with replacing a checkpoint.
    original = path.read_bytes()
    path.write_bytes(original[:-1] + b"B")
    changed = benchmark.reference_signature(args, 12, 16, 12, 1)
    assert input_name not in saved and input_name not in changed
    assert len(original) == path.stat().st_size
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, changed)


@pytest.mark.parametrize("version", [None, 1, 2, 4])
def test_incomplete_or_incompatible_protocol_requires_new_reference(
    benchmark, args, version
):
    current = benchmark.reference_signature(args, 12, 16, 12, 1)
    saved = dict(current)
    if version is None:
        saved.pop("protocol_version")
    else:
        saved["protocol_version"] = version
    with pytest.raises(ValueError, match="protocol version differs; regenerate"):
        benchmark.validate_reference_signature({"signature": saved}, current)


def test_missing_workload_setting_is_not_silently_accepted(benchmark, args):
    current = benchmark.reference_signature(args, 12, 16, 12, 1)
    saved = dict(current)
    saved.pop("compact_box")
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, current)


@pytest.mark.parametrize(
    ("rest_names", "omit", "error"),
    [
        (None, ("scale_1",), "missing required vertex properties"),
        (["f_rest_0", "f_rest_1", "f_rest_3"], (), "contiguous from f_rest_0"),
        (["f_rest_0", "f_rest_1"], (), "three f_rest coefficients per SH basis"),
        ([f"f_rest_{index}" for index in range(6)], (), "basis count"),
        (
            [f"f_rest_{index}" for index in range(105)],
            (),
            "supports degrees 0 through 4",
        ),
        (["f_rest_invalid"], (), "must end in an integer"),
    ],
)
def test_scene_metadata_rejects_invalid_3dgs_schema(
    benchmark, tmp_path, rest_names, omit, error
):
    ply, cameras = tmp_path / "invalid.ply", tmp_path / "cameras.json"
    write_scene_files(ply, cameras, rest_names=rest_names, omit=omit)
    with pytest.raises(ValueError, match=error):
        benchmark.inspect_scene_metadata(ply, cameras, 0, 0)


def test_malformed_ply_rejected_before_extension_and_cuda(
    benchmark, args, tmp_path, monkeypatch
):
    args.ply.write_text(args.ply.read_text().replace("f_rest_8", "f_rest_bad"))
    args.extension = tmp_path / "missing.so"
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Malformed PLY reached extension loading or tensor loading")

    monkeypatch.setattr(benchmark.importlib.util, "spec_from_file_location", unexpected)
    monkeypatch.setattr(benchmark, "load_scene", unexpected)
    with pytest.raises(
        ValueError, match="f_rest property names must end in an integer"
    ):
        benchmark.main()


def test_derived_scene_mismatch_rejects_before_extension_and_gpu(
    benchmark, args, tmp_path, monkeypatch
):
    signature = benchmark.reference_signature(args, 11, 16, 12, 1)
    signature["points_total"] = 12
    reference_path = tmp_path / "derived-mismatch.pt"
    import torch

    torch.save({"signature": signature}, reference_path)
    args.compare_reference = reference_path
    args.extension = tmp_path / "missing.so"
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Derived-metadata mismatch reached extension or scene loading")

    monkeypatch.setattr(benchmark.importlib.util, "spec_from_file_location", unexpected)
    monkeypatch.setattr(benchmark, "load_scene", unexpected)
    with pytest.raises(ValueError, match="input contents or rendering settings differ"):
        benchmark.main()


def test_relocated_identical_inputs_match(benchmark, args, tmp_path):
    saved = benchmark.reference_signature(args, 12, 16, 12, 1)
    provenance = benchmark.reference_provenance(args)
    for name in ("ply", "cameras"):
        old = getattr(args, name)
        relocated = tmp_path / ("relocated-" + old.name)
        relocated.write_bytes(old.read_bytes())
        setattr(args, name, relocated)
    benchmark.validate_reference_signature(
        {"signature": saved}, benchmark.reference_signature(args, 12, 16, 12, 1)
    )
    assert provenance != benchmark.reference_provenance(args)


@pytest.mark.parametrize(
    "failure", ["legacy", "bytes", "settings", "derived_metadata", "matching"]
)
def test_main_cpu_preflight(benchmark, args, tmp_path, monkeypatch, failure):
    import torch

    saved = benchmark.reference_signature(args, 12, 16, 12, 1)
    saved["points_total"] = 12
    reference = tmp_path / "reference.pt"
    if failure == "legacy":
        saved["protocol_version"] = 2
        args.ply.unlink()
        args.cameras.unlink()
    elif failure == "bytes":
        args.ply.write_bytes(b"changed")
    elif failure == "settings":
        args.tile_size = 8
    elif failure == "derived_metadata":
        saved["points_used"] = 11
    torch.save({"signature": saved}, reference)
    args.compare_reference = reference
    args.extension = tmp_path / "missing.so"
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    class ReachedExtension(Exception):
        pass

    def extension_guard(*_args, **_kwargs):
        if failure == "matching":
            raise ReachedExtension
        pytest.fail("Incompatible reference reached extension loading")

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU preflight must not upload a scene or initialize CUDA")

    monkeypatch.setattr(
        benchmark.importlib.util, "spec_from_file_location", extension_guard
    )
    monkeypatch.setattr(benchmark, "load_scene", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "Generator", forbidden)
    with pytest.raises(ReachedExtension if failure == "matching" else ValueError):
        benchmark.main()


@pytest.mark.parametrize("field", ["points_used", "resolution", "degree"])
def test_derived_metadata_must_match(benchmark, args, field):
    saved = benchmark.reference_signature(args, 12, 1280, 720, 1)
    current = dict(saved)
    current[field] = -1
    with pytest.raises(ValueError, match="rendering settings differ"):
        benchmark.validate_reference_signature({"signature": saved}, current)
