import inspect

import pytest


PUBLIC_FUNCS = [
    ("gsplat.rendering", "rasterization"),
    ("gsplat.rendering", "rasterization_2dgs"),
    ("gsplat.cuda._wrapper", "fully_fused_projection"),
    ("gsplat.cuda._wrapper", "fully_fused_projection_2dgs"),
]


@pytest.mark.parametrize(("module_name", "func_name"), PUBLIC_FUNCS)
def test_public_rasterizers_default_packed_false(module_name: str, func_name: str):
    module = __import__(module_name, fromlist=[func_name])
    func = getattr(module, func_name)
    sig = inspect.signature(func)
    param = sig.parameters["packed"]
    assert param.default is False, (
        f"{module_name}.{func_name}.packed default is {param.default!r}; "
        "expected False so the dense strategy is the default."
    )
