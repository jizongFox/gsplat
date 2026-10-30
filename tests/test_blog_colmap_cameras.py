import importlib.util
from pathlib import Path

import numpy as np
import pytest


def converter():
    path = Path(__file__).parents[1] / "benchmarks/blog_eval/colmap_cameras.py"
    spec = importlib.util.spec_from_file_location("colmap_cameras", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def test_colmap_pose_direction_and_image_order(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 1280 720 560 559 635 362\n"
    )
    (tmp_path / "images.txt").write_text(
        "# images\n9 1 0 0 0 1 2 3 1 b.jpeg\n\n"
        "5 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 a.jpeg\n"
        "0 0 -1\n"
    )
    cameras = converter()(tmp_path)
    assert [c["id"] for c in cameras] == [5, 9]
    assert cameras[0]["cx"] == 635
    assert cameras[0]["cy"] == 362
    np.testing.assert_allclose(cameras[0]["position"], [-2, 1, -3], atol=1e-14)
    np.testing.assert_allclose(
        cameras[0]["rotation"], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-14
    )
    np.testing.assert_allclose(cameras[1]["position"], [-1, -2, -3])


@pytest.mark.parametrize("model", ["SIMPLE_PINHOLE", "OPENCV"])
def test_reject_non_pinhole(tmp_path, model):
    (tmp_path / "cameras.txt").write_text(f"1 {model} 1280 720 560 559 635 362\n")
    with pytest.raises(ValueError, match="PINHOLE"):
        converter()(tmp_path)


def test_reject_missing_observations_line(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 PINHOLE 1280 720 560 559 635 362\n")
    (tmp_path / "images.txt").write_text("5 1 0 0 0 0 0 0 1 image.jpeg\n")
    with pytest.raises(ValueError, match="observations"):
        converter()(tmp_path)
