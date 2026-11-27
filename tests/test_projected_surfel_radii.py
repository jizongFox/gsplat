import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projected_surfel_radii_matches_expected_values():
    from gsplat import projected_surfel_radii

    transforms = torch.tensor(
        [
            [[3.0, 0.0, 20.0], [4.0, 0.0, 20.0], [0.0, 0.0, 1.0]],
            [[23.998, 0.0, 20.0], [19.998, 2.0, 20.0], [0.9999, 0.0, 1.0]],
            [[1.0, 0.0, -100.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        ],
        device="cuda",
    )
    raw_radii = torch.tensor([12, 1_000_000, 1_000_000], device="cuda", dtype=torch.int32)
    actual = projected_surfel_radii(transforms, raw_radii, 40, 40)
    assert actual.tolist() == [15, 13, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projected_surfel_radii_handles_empty_input():
    from gsplat import projected_surfel_radii

    transforms = torch.empty((0, 3, 3), device="cuda")
    raw_radii = torch.empty((0,), device="cuda", dtype=torch.int32)
    assert projected_surfel_radii(transforms, raw_radii, 40, 40).shape == (0,)
