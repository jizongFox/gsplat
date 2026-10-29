"""CPU-only checks that resolution summaries preserve parity failures."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def summary_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/blog_eval/summarize_megapixels.py"
    )
    spec = importlib.util.spec_from_file_location("mp_summary_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(error=0.0, passed=True):
    return {
        "finite": True,
        "max_abs": error,
        "rel_l2": error,
        "alignment_passed": passed,
    }


def write_jobs(directory):
    for variant, scale in (("base", 1.0), ("pr", 0.5)):
        records = []
        for scope in ("full", "raster"):
            for camera in (0, 1, 2):
                factor = scale * (camera + 1)
                image = {
                    **metric(),
                    "psnr": None,
                    "rgb_max_abs": 0.0,
                    "depth_max_abs": 0.0,
                }
                failed = scope == "full" and camera == 2
                records.append(
                    {
                        "scope": scope,
                        "resolution": [640, 360],
                        "camera_index": camera,
                        "status": "ok",
                        "finite_checks": {"training": {"image": True}},
                        "timing": {
                            "training_forward": {"mean_ms": 10 * factor},
                            "backward": {"mean_ms": 20 * factor},
                            "combined": {"mean_ms": 30 * factor},
                            "forward_only": {"mean_ms": 8 * factor},
                        },
                        "parity": (
                            {
                                "image": image,
                                "alpha": metric(),
                                "grad.quats": metric(
                                    1e-3 if failed else 1e-6, not failed
                                ),
                            }
                            if variant == "pr"
                            else None
                        ),
                        "baseline_repeat_noise": {"grad.quats": metric(1e-5)},
                        "baseline_provenance": {"extension_sha256": "base"},
                        "repeat_noise": {"grad.quats": metric(2e-5)},
                    }
                )
        report = {
            "inputs": {"ply": "/scene.ply", "cameras": "/cameras.json", "hash": "abc"},
            "config": {
                "skip_validation": variant == "base",
                "warmup": 20,
                "scope": "both",
                "resolutions": [[640, 360]],
                "camera_indices": [0, 1, 2],
            },
            "extension": {"sha256": variant},
            "records": records,
        }
        (directory / f"{variant}-job1.json").write_text(json.dumps(report))


def test_summary_keeps_failed_gradient_and_correct_aggregation(tmp_path, monkeypatch):
    write_jobs(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["summary", "--directory", str(tmp_path), "--jobs", "1"]
    )
    summary_module().main()
    result = json.loads((tmp_path / "parity_summary.json").read_text())
    assert result["forward_passes"] == 6
    assert result["gradient_case_passes"] == 5
    assert len(result["gradient_failures"]) == 1
    assert result["gradient_failures"][0]["camera_index"] == 2
    assert result["gradient_failures"][0]["rel_l2"] == 1e-3
    with (tmp_path / "overview.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert float(row["base_combined_ms"]) == 60
        assert float(row["pr_combined_ms"]) == 30
        assert float(row["speedup_combined_ms"]) == 2
    with (tmp_path / "quality.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(row["gradient_passed"] == "False" for row in rows) == 1


@pytest.mark.parametrize(
    "change",
    ["settings", "contents", "timing_sum", "missing_case", "oom", "wrong_reference"],
)
def test_summary_rejects_inconsistent_workloads(tmp_path, monkeypatch, change):
    write_jobs(tmp_path)
    path = tmp_path / "pr-job1.json"
    report = json.loads(path.read_text())
    if change == "settings":
        report["config"]["warmup"] = 0
    elif change == "contents":
        report["inputs"]["hash"] = "different"
    elif change == "timing_sum":
        report["records"][0]["timing"]["combined"]["mean_ms"] = 999
    elif change == "missing_case":
        report["records"].pop()
    elif change == "oom":
        report["records"][0]["status"] = "oom"
    elif change == "wrong_reference":
        report["records"][0]["baseline_provenance"]["extension_sha256"] = "oldbinary"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        sys, "argv", ["summary", "--directory", str(tmp_path), "--jobs", "1"]
    )
    with pytest.raises(ValueError):
        summary_module().main()
