"""CPU checks for fixture identity and independent-traversal coverage."""

import hashlib
import importlib.util
import json
from pathlib import Path


def test_reference_cases_preserved_and_warp_cases_appended():
    source = (
        Path(__file__).parents[1] / "benchmarks/blog_eval/gsplat_raster_regression.py"
    )
    spec = importlib.util.spec_from_file_location("warp_private_fixtures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.cases()
    digest = hashlib.sha256(
        json.dumps(cases[:166], sort_keys=True).encode()
    ).hexdigest()
    assert digest == "98f9aaba737f37d0291e825c787101ff60635d9b44d214f1c81bbce796355a91"
    assert len(cases) == 193
    assert len({case["name"] for case in cases}) == len(cases)
    added = cases[166:]
    assert any(case["scenario"] == "warp_imbalance" for case in added)
    assert any(case["scenario"] == "warp_staging_boundary" for case in added)
    assert any(case["height"] == 1 for case in added)
    assert any(case["tile_size"] == 4 and case["channels"] == 7 for case in added)
    assert any(case["tile_size"] == 6 and case["channels"] == 7 for case in added)
    assert any(case["channels"] == 9 for case in added)
