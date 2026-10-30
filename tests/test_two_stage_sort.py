"""Exact public intersection-output contract, including original stable ties."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture(cameras=1, count=129, tile_width=80, tile_height=45, packed=False):
    generator = torch.Generator().manual_seed(425)
    means = torch.rand((cameras, count, 2), generator=generator)
    means *= torch.tensor([tile_width * 16, tile_height * 16])
    radii = torch.randint(
        0, 49, (cameras, count), generator=generator, dtype=torch.int32
    )
    # Many exactly equal depths; keep original flatten order as the tie breaker.
    depths = torch.randint(1, 8, (cameras, count), generator=generator).float()
    conics = torch.zeros((cameras, count, 3))
    conics[..., 0] = 0.05
    conics[..., 2] = 0.08
    opacity = torch.full((cameras, count), 0.5)
    if count:
        opacity[:, ::7] = 0.001  # Positive radius, but no compact-box intersections.
    data = dict(means2d=means, radii=radii, depths=depths)
    options = dict(tile_size=16, tile_width=tile_width, tile_height=tile_height)
    if packed:
        # Interleave cameras rather than assuming the input is camera-major.
        data = {
            key: value.transpose(0, 1).reshape(-1, *value.shape[2:])
            for key, value in data.items()
        }
        conics = conics.transpose(0, 1).reshape(-1, 3)
        opacity = opacity.transpose(0, 1).reshape(-1)
        options.update(
            packed=True,
            n_cameras=cameras,
            camera_ids=torch.arange(cameras).repeat(count).cuda(),
            gaussian_ids=torch.arange(count).repeat_interleave(cameras).cuda(),
        )
    data = {key: value.cuda().contiguous() for key, value in data.items()}
    options.update(
        conics=conics.cuda().contiguous(), opacities=opacity.cuda().contiguous()
    )
    return data, options


def compare(data, options, independent_order=True):
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode

    baseline = isect_tiles(**data, **options)
    candidate = isect_tiles(**data, **options, two_stage_sort=True)
    assert baseline[0].dtype == candidate[0].dtype == torch.int32
    assert baseline[1].dtype == candidate[1].dtype == torch.int64
    assert baseline[2].dtype == candidate[2].dtype == torch.int32
    assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
    if independent_order and options.get("sort", True):
        unsorted = isect_tiles(**data, **{**options, "sort": False})
        order = torch.argsort(unsorted[1], stable=True)
        assert torch.equal(candidate[1], unsorted[1][order])
        assert torch.equal(candidate[2], unsorted[2][order])
    # Invalid live depth bits can corrupt the *baseline* encoded namespace;
    # those fallback tests promise identical intersection bytes, not valid
    # downstream rendering/offsets for malformed depth inputs.
    if independent_order and options.get("sort", True):
        cameras = options.get("n_cameras", data["depths"].shape[0])
        args = (cameras, options["tile_width"], options["tile_height"])
        assert torch.equal(
            isect_offset_encode(baseline[1], *args),
            isect_offset_encode(candidate[1], *args),
        )
    return baseline, candidate


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compact", ["off", "rect_min", "sweep"])
@pytest.mark.parametrize("count", [0, 1, 129, 513])
def test_exact_keys_ids_and_ties(packed, compact, count):
    data, options = fixture(cameras=2, count=count, packed=packed)
    if compact != "off":
        options.update(compact_box=True, compact_box_impl=compact)
    compare(data, options)


@pytest.mark.parametrize(
    "cameras,width,height", [(1, 256, 256), (2, 255, 128), (2, 256, 128), (3, 200, 100)]
)
def test_encoded_camera_namespace_boundaries(cameras, width, height):
    # C=3, 20000 tiles: product60000 fits, but encoded camera stride makes
    # camera2 start at65536. A product-only 16-bit guard would truncate IDs.
    data, options = fixture(cameras, count=8, tile_width=width, tile_height=height)
    compare(data, options)


@pytest.mark.parametrize(
    "value", [-1.0, -0.0, 0.0, float("inf"), float("-inf"), float("nan")]
)
@pytest.mark.parametrize("visible", [False, True])
def test_nonpositive_or_nonfinite_depth_fallback(value, visible):
    data, options = fixture(count=17)
    data["depths"][0, 0] = value
    data["radii"][0, 0] = 16 if visible else 0
    compare(data, options, independent_order=False)


def test_neighboring_float_depth_bits_and_all_ties():
    data, options = fixture(count=128)
    data["means2d"][:] = torch.tensor([15.5, 15.5], device="cuda")
    data["radii"].fill_(16)
    data["depths"].fill_(1.0)
    compare(data, options)
    data["depths"][:, ::2] = torch.nextafter(
        data["depths"][:, ::2], torch.full_like(data["depths"][:, ::2], float("inf"))
    )
    compare(data, options)


def test_unsorted_contract_unchanged():
    data, options = fixture(count=33, packed=True)
    options["sort"] = False
    compare(data, options)


def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data, options = fixture(cameras=2, count=513, packed=True)
        options.update(compact_box=True, compact_box_impl="sweep")
        compare(data, options)
    stream.synchronize()


def test_gaussian_ids_above_24_bits_are_not_packed_or_truncated():
    # Keep only the final Gaussian visible, with an ID that cannot fit in24bits.
    count = 2**24 + 9
    data = dict(
        means2d=torch.zeros((1, count, 2), device="cuda"),
        radii=torch.zeros((1, count), device="cuda", dtype=torch.int32),
        depths=torch.zeros((1, count), device="cuda"),
    )
    data["means2d"][0, -1] = 8.0
    data["radii"][0, -1] = 1
    data["depths"][0, -1] = 1.0
    _, candidate = compare(data, dict(tile_size=16, tile_width=2, tile_height=2))
    assert candidate[2].tolist() == [count - 1]


def test_float64_uses_compatible_fallback():
    data, options = fixture(count=17)
    data = {k: (v.double() if v.is_floating_point() else v) for k, v in data.items()}
    options = {
        k: (v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in options.items()
    }
    compare(data, options, independent_order=False)


@pytest.mark.parametrize("double_buffer", [False, True])
def test_cpp_buffer_modes_match_original_entry(double_buffer):
    from gsplat.cuda._backend import _C

    data, options = fixture(cameras=2, count=513, packed=True)
    args = (
        data["means2d"],
        data["radii"],
        data["depths"],
        options["conics"],
        None,
        options["opacities"],
        options["camera_ids"],
        options["gaussian_ids"],
        2,
        16,
        80,
        45,
        True,
        1.0,
        0.0,
        False,
        True,
        True,
        double_buffer,
    )
    original = _C.isect_tiles(*args)
    candidate = _C.isect_tiles_two_stage(*args)
    assert all(torch.equal(a, b) for a, b in zip(original, candidate))
