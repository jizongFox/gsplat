"""The serial job driver must not silently accept failed worker records."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", [None, "exit", "parity", "oom", "finite"])
def test_followup_driver(tmp_path, monkeypatch, failure):
    path = Path(__file__).parents[1] / "benchmarks/blog_eval/run_followup.py"
    spec = importlib.util.spec_from_file_location("followup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "followup",
            "--baseline",
            "base.so",
            "--candidate",
            "candidate.so",
            "--ply",
            "scene.ply",
            "--cameras",
            "cameras.json",
            "--camera-indices",
            "0",
            "--output-dir",
            str(output),
            "--reference-dir",
            str(tmp_path / "refs"),
            "--jobs",
            "3",
        ],
    )
    monkeypatch.setattr(module, "telemetry", lambda: "test GPU")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        record = {
            "camera_index": 0,
            "resolution": [640, 360],
            "scope": "full",
            "status": "oom" if failure == "oom" else "ok",
            "alignment_passed": failure != "parity",
            "finite_checks": {"training": {"image": failure != "finite"}},
        }
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"records": [record]})
        )
        return SimpleNamespace(returncode=int(failure == "exit"))

    monkeypatch.setattr(module.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError):
            module.main()
        assert len(commands) == (6 if failure == "parity" else 1)
    else:
        module.main()
        assert [c[c.index("--label") + 1] for c in commands] == [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        ]
        assert "--save-reference-dir" in commands[0]
        assert "--compare-reference-dir" in commands[1]
        assert all("--skip-validation" in c for c in commands[2:])
        # Preserve compatibility with PR9's existing summarizer/plot tools.
        assert (output / "base-job1.json").is_file()
        assert (output / "pr-job3.json").is_file()


def test_pr9_fixture_order_and_seeds_preserved():
    path = (
        Path(__file__).parents[1] / "benchmarks/blog_eval/gsplat_raster_regression.py"
    )
    spec = importlib.util.spec_from_file_location("raster_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.cases()
    # Canonical config hash from a7050c6: each fixture's seed includes its index.
    digest = hashlib.sha256(
        json.dumps(cases[:166], sort_keys=True).encode()
    ).hexdigest()
    assert digest == "98f9aaba737f37d0291e825c787101ff60635d9b44d214f1c81bbce796355a91"
    assert len(cases) == 174
    assert all(c["tile_size"] == 32 and c["count"] > 1024 for c in cases[166:])
