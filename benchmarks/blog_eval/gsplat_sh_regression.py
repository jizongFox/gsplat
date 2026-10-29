"""Differential SH wrapper checks, with one extension per isolated worker.

Use identical Python/PYTHONPATH for baseline and candidate::

    python benchmarks/gsplat_sh_regression.py --extension baseline.so --save base.pt
    python benchmarks/gsplat_sh_regression.py --extension candidate.so \
        --compare base.pt --output report.json

Masked forward rows are undefined (the kernel skips writes), so only active
rows are compared. Full gradients must be finite, with exact zeros for masked
rows and unused coefficients. This script runs CUDA only in its worker process.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import typing as t
from pathlib import Path


def cases(repeats: int = 1) -> t.List[t.Dict[str, t.Any]]:
    result = []
    for repeat in range(repeats):
        for degree in range(5):
            for bases in sorted({(degree + 1) ** 2, 25}):
                for count in (0, 1, 31, 257):
                    for mask in ("none", "alltrue", "allfalse", "random"):
                        for dirs_grad in (False, True):
                            result.append(
                                dict(
                                    name=(
                                        f"r{repeat}_d{degree}_k{bases}_n{count}_"
                                        f"{mask}_dirsgrad{int(dirs_grad)}"
                                    ),
                                    degree=degree,
                                    bases=bases,
                                    count=count,
                                    mask=mask,
                                    dirs_grad=dirs_grad,
                                )
                            )
    return result


def run_case(config: t.Dict[str, t.Any], seed: int, wrapper: t.Any) -> t.Tuple:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    count, bases = config["count"], config["bases"]
    directions = torch.randn(count, 3, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    coeffs = torch.randn(count, bases, 3, generator=generator)
    upstream = torch.randn(count, 3, generator=generator)
    mask = None
    if config["mask"] == "random":
        mask = torch.rand(count, generator=generator) > 0.5
        # Ensure both branches occur in every nontrivial random-mask fixture.
        if count > 1:
            mask[0], mask[-1] = True, False
    elif config["mask"] != "none":
        mask = torch.full((count,), config["mask"] == "alltrue", dtype=torch.bool)
    active = torch.ones(count, dtype=torch.bool) if mask is None else mask
    dirs = directions.cuda().requires_grad_(config["dirs_grad"])
    coefficients = coeffs.cuda().requires_grad_()
    colors = wrapper.spherical_harmonics(
        config["degree"], dirs, coefficients, None if mask is None else mask.cuda()
    )
    # Use explicit upstream gradients, never a loss reading undefined masked rows.
    colors.backward(upstream.cuda())
    if coefficients.grad is None:
        raise AssertionError("Missing coefficient gradient")
    tensors = {
        "colors/active": colors.detach().cpu()[active],
        "grad/coeffs": coefficients.grad.detach().cpu(),
    }
    if config["dirs_grad"]:
        if dirs.grad is None:
            raise AssertionError("Missing direction gradient")
        tensors["grad/dirs"] = dirs.grad.detach().cpu()
    elif dirs.grad is not None:
        raise AssertionError("Unexpected direction gradient")
    checks = {
        "masked_coeffs_zero": bool((tensors["grad/coeffs"][~active] == 0).all()),
        "unused_coeffs_zero": bool(
            (tensors["grad/coeffs"][:, (config["degree"] + 1) ** 2 :] == 0).all()
        ),
    }
    if config["dirs_grad"]:
        checks["masked_dirs_zero"] = bool((tensors["grad/dirs"][~active] == 0).all())
        if config["degree"] == 0:
            checks["degree_zero_dirs_zero"] = bool((tensors["grad/dirs"] == 0).all())
    return tensors, checks


def compare_tensors(actual: t.Dict, expected: t.Dict) -> t.Dict:
    import torch

    if actual.keys() != expected.keys():
        raise ValueError("Reference tensor names differ")
    results = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Reference shape/dtype differs: {name}")
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        delta = value.double() - reference.double()
        rel_l2 = (
            float(delta.norm() / reference.double().norm().clamp_min(1e-30))
            if finite
            else None
        )
        results[name] = dict(
            finite=finite,
            max_abs=(
                (float(delta.abs().max()) if delta.numel() else 0.0) if finite else None
            ),
            rel_l2=rel_l2,
            passed=finite and rel_l2 <= 1e-4,
        )
    return results


def worker(args: argparse.Namespace) -> int:
    import gsplat
    import torch

    # Inject BEFORE _backend: avoid installed/JIT binaries and duplicate pybind
    # registrations. The parent process never imports torch or gsplat.
    spec = importlib.util.spec_from_file_location(
        "gsplat.csrc", args.extension.resolve()
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load extension: {args.extension}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat.csrc"] = extension
    gsplat.csrc = extension
    from gsplat.cuda import _backend, _wrapper

    if _backend._C is not extension:
        raise RuntimeError("gsplat backend did not use the requested extension")
    configs = cases(args.repeats)
    reference = None
    if args.compare:
        reference = torch.load(args.compare, map_location="cpu", weights_only=True)
        if (
            reference["version"] != 1
            or reference["seed"] != args.seed
            or reference["configs"] != configs
        ):
            raise ValueError("Reference suite version, seed, or configuration differs")
        if set(reference["tensors"]) != {config["name"] for config in configs}:
            raise ValueError("Reference case names differ")
        if not reference["passed"]:
            raise ValueError("Reference failed its finite/zero-gradient checks")
    report = dict(
        extension=str(args.extension.resolve()),
        reference=str(args.compare) if args.compare else None,
        seed=args.seed,
        thresholds=dict(rel_l2=1e-4),
        masked_forward="undefined; compare active rows only",
        cases={},
        passed=True,
    )
    saved = {}
    for index, config in enumerate(configs):
        name = config["name"]
        actual, checks = run_case(config, args.seed + index, _wrapper)
        expected = reference["tensors"][name] if reference else actual
        metrics = compare_tensors(actual, expected)
        passed = all(checks.values()) and all(m["passed"] for m in metrics.values())
        report["cases"][name] = dict(passed=passed, checks=checks, tensors=metrics)
        report["passed"] = report["passed"] and passed
        if args.save:
            saved[name] = actual
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                version=1,
                seed=args.seed,
                configs=configs,
                tensors=saved,
                passed=report["passed"],
            ),
            args.save,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(configs)} cases: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--save", type=Path)
    action.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.extension.is_file():
        parser.error(f"Extension does not exist: {args.extension}")
    if args.compare and not args.compare.is_file():
        parser.error(f"Reference does not exist: {args.compare}")
    if args.worker:
        return worker(args)
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
