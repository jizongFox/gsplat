"""CPU-only identity and early-preflight checks for the fixed-count ensemble."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def ensemble():
    directory = Path(__file__).resolve().parents[1] / "benchmarks/blog_eval"
    # The executable imports the adjacent single-pass scene loader.
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "blog_ensemble_under_test", directory / "gsplat_gradient_ensemble.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


@pytest.fixture
def args(ensemble, tmp_path):
    ply = tmp_path / "scene.ply"
    cameras = tmp_path / "cameras.json"
    properties = ["x", "y", "z"]
    properties += [f"scale_{i}" for i in range(3)]
    properties += [f"rot_{i}" for i in range(4)]
    properties += ["opacity"]
    properties += [f"f_dc_{i}" for i in range(3)]
    properties += [f"f_rest_{i}" for i in range(9)]
    rows = []
    for index in range(12):
        values = [index * 0.01, 0.0, 3.0]
        values += [-2.0, -2.0, -2.0]
        values += [1.0, 0.0, 0.0, 0.0]
        values += [0.0]
        values += [0.1, 0.2, 0.3]
        values += [0.0] * 9
        rows.append(" ".join(map(str, values)))
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
    return ensemble.parse_args(
        [
            "--ply",
            str(ply),
            "--cameras",
            str(cameras),
            "--extension",
            str(tmp_path / "missing.so"),
            "--compare-reference",
            str(tmp_path / "baseline.pt"),
        ]
    )


def reference(ensemble, args):
    metadata = ensemble.inspect_scene_metadata(
        args.ply, args.cameras, args.max_points, args.camera_index
    )
    signature = ensemble.reference_input_signature(args)
    signature.update(
        points_used=metadata["points_used"],
        points_total=metadata["points_total"],
        resolution=metadata["resolution"],
        degree=metadata["degree"],
    )
    return {
        "signature": signature,
        "repeats": args.repeats,
    }


def test_matching_and_relocated_inputs(ensemble, args, tmp_path):
    saved = reference(ensemble, args)
    provenance = ensemble.reference_provenance(args)
    for name in ("ply", "cameras"):
        original = getattr(args, name)
        relocated = tmp_path / ("moved-" + original.name)
        relocated.write_bytes(original.read_bytes())
        setattr(args, name, relocated)
    current = reference(ensemble, args)["signature"]
    ensemble.validate_reference_signature(saved, current)
    assert "ply" not in current and "cameras" not in current
    assert provenance != ensemble.reference_provenance(args)
    assert current["script_sha256"] == ensemble.file_hash(Path(ensemble.__file__))
    assert current["protocol"] == "predeclared_raw_parameter_mean_v2"
    assert "protocol_version" not in current


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "ply",
        "cameras",
        "mode",
        "repeats",
        "camera_index",
        "max_points",
        "matching",
        "derived_metadata",
    ],
)
def test_main_preflight_before_extension_scene_or_cuda(
    ensemble, args, monkeypatch, change
):
    saved = reference(ensemble, args)
    if change == "legacy":
        saved["signature"]["protocol"] = "predeclared_raw_parameter_mean_v1"
        args.ply.unlink()
        args.cameras.unlink()
    elif change in ("ply", "cameras"):
        getattr(args, change).write_bytes(b"changed")
    elif change == "mode":
        args.mode = "RGB+ED"
    elif change == "derived_metadata":
        saved["signature"]["points_used"] += 1
    elif change != "matching":
        setattr(args, change, getattr(args, change) + 1)
    torch.save(saved, args.compare_reference)
    monkeypatch.setattr(ensemble, "parse_args", lambda: args)

    class ReachedExtension(Exception):
        pass

    def extension_guard(*_args, **_kwargs):
        if change == "matching":
            raise ReachedExtension
        pytest.fail("Incompatible reference reached extension loading")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight must not upload a scene or initialize CUDA")

    monkeypatch.setattr(
        ensemble.importlib.util, "spec_from_file_location", extension_guard
    )
    monkeypatch.setattr(ensemble, "load_scene", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    monkeypatch.setattr(torch, "Generator", forbidden)
    with pytest.raises(ReachedExtension if change == "matching" else ValueError):
        ensemble.main()


def test_reference_actions_exclusive(ensemble, args):
    with pytest.raises(SystemExit) as error:
        ensemble.parse_args(
            [
                "--ply",
                str(args.ply),
                "--cameras",
                str(args.cameras),
                "--extension",
                str(args.extension),
                "--compare-reference",
                str(args.compare_reference),
                "--save-reference",
                str(args.compare_reference),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field", ["points_used", "points_total", "resolution", "degree"]
)
def test_derived_metadata_checked_in_second_stage(ensemble, args, field):
    saved = reference(ensemble, args)
    saved["signature"].update(
        points_used=12, points_total=20, resolution=[1280, 720], degree=1
    )
    current = dict(saved["signature"])
    current[field] = -1
    with pytest.raises(ValueError, match="signature differs"):
        ensemble.validate_reference_signature(saved, current, partial=True)
