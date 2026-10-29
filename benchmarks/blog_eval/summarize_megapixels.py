"""Flatten serial megapixel worker reports into plotting CSV and parity tables."""

import argparse
import csv
import json
import math
import re
import statistics
import typing as t
from pathlib import Path

STAGES = {
    "forward_ms": "training_forward",
    "backward_ms": "backward",
    "combined_ms": "combined",
    "inference_ms": "forward_only",
}


def write_csv(path: Path, rows: t.List[t.Dict[str, t.Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    timings, quality, gradient_rows, failures, skipped = [], [], [], [], []
    workloads, hashes = {}, {}
    expected_files = {
        f"{variant}-job{job}.json"
        for variant in ("base", "pr")
        for job in range(1, args.jobs + 1)
    }
    if not all((args.directory / name).is_file() for name in expected_files):
        raise ValueError("Incomplete baseline/candidate job coverage")
    for path in sorted(args.directory.glob("*-job*.json")):
        match = re.fullmatch(r"(base|pr)-job(\d+)\.json", path.name)
        if match is None:
            continue
        variant, job = match[1], int(match[2])
        if path.name not in expected_files:
            raise ValueError(f"Unexpected timing job: {path}")
        report = json.loads(path.read_text())
        config = report["config"]
        scopes = ("full", "raster") if config["scope"] == "both" else (config["scope"],)
        expected_cases = {
            (scope, width, height, camera)
            for scope in scopes
            for width, height in config["resolutions"]
            for camera in config["camera_indices"]
        }
        actual_cases = [
            (p["scope"], *p["resolution"], p["camera_index"]) for p in report["records"]
        ]
        if (
            len(actual_cases) != len(expected_cases)
            or set(actual_cases) != expected_cases
        ):
            raise ValueError(f"Incomplete or duplicate case coverage: {path}")
        identity = {**report["inputs"]}
        identity.pop("ply")
        identity.pop("cameras")
        workload = {
            "inputs": identity,
            "config": {
                key: value
                for key, value in report["config"].items()
                if key != "skip_validation"
            },
        }
        if workloads and workload != next(iter(workloads.values())):
            raise ValueError(f"Worker workload mismatch: {path}")
        workloads[path.name] = workload
        digest = report["extension"]["sha256"]
        if variant in hashes and hashes[variant] != digest:
            raise ValueError(f"Binary changed between jobs for {variant}")
        hashes[variant] = digest
        for point in report["records"]:
            width, height = point["resolution"]
            key = dict(
                scope=point["scope"],
                width=width,
                height=height,
                camera_index=point["camera_index"],
            )
            if point["status"] != "ok":
                raise ValueError(
                    f"Cannot aggregate skipped/OOM cases as equal workloads: {path}, {key}"
                )
            if not all(
                all(checks.values()) for checks in point["finite_checks"].values()
            ):
                raise ValueError(f"Nonfinite timed outputs/gradients: {path}, {key}")
            values = {
                column: point["timing"][stage]["mean_ms"]
                for column, stage in STAGES.items()
            }
            if not math.isclose(
                values["forward_ms"] + values["backward_ms"],
                values["combined_ms"],
                abs_tol=1e-3,
            ):
                raise ValueError(f"Training timing components do not add: {key}")
            timings.append({**key, "variant": variant, "job": job, **values})
            if variant != "pr" or point["parity"] is None:
                continue
            parity, noise = point["parity"], point["baseline_repeat_noise"]
            if point["baseline_provenance"]["extension_sha256"] != hashes["base"]:
                raise ValueError("Parity reference binary differs from timing baseline")
            gradients = {k: v for k, v in parity.items() if k.startswith("grad.")}
            if not gradients or noise is None:
                raise ValueError(f"Missing gradient comparison/repeat noise: {key}")
            if not all(value["finite"] for value in parity.values()):
                raise ValueError(f"Nonfinite parity comparison: {key}")
            gradient_passed = all(v["alignment_passed"] for v in gradients.values())
            forward_passed = all(
                parity[k]["alignment_passed"] for k in ("image", "alpha")
            )
            quality.append(
                {
                    **key,
                    "rgb_psnr_db": parity["image"]["psnr"],
                    "rgb_exact": parity["image"]["rgb_max_abs"] == 0,
                    "image_max_abs": parity["image"]["max_abs"],
                    "alpha_max_abs": parity["alpha"]["max_abs"],
                    "depth_max_abs": parity["image"]["depth_max_abs"],
                    "grad_max_rel_l2": max(v["rel_l2"] for v in gradients.values()),
                    "baseline_noise_max_rel_l2": max(
                        noise[k]["rel_l2"] for k in gradients
                    ),
                    "gradient_passed": gradient_passed,
                    "forward_passed": forward_passed,
                }
            )
            for name, metrics in gradients.items():
                gradient_rows.append(
                    {
                        **key,
                        "gradient": name,
                        "rel_l2": metrics["rel_l2"],
                        "max_abs": metrics["max_abs"],
                        "baseline_repeat_rel_l2": noise[name]["rel_l2"],
                        "candidate_repeat_rel_l2": point["repeat_noise"][name][
                            "rel_l2"
                        ],
                        "passed": metrics["alignment_passed"],
                    }
                )
                if not metrics["alignment_passed"]:
                    failures.append(gradient_rows[-1])
    write_csv(args.directory / "timings.csv", timings)
    write_csv(args.directory / "quality.csv", quality)
    write_csv(args.directory / "gradients.csv", gradient_rows)
    summary = {
        "binary_sha256": hashes,
        "timing_rows": len(timings),
        "quality_cases": len(quality),
        "forward_passes": sum(row["forward_passed"] for row in quality),
        "gradient_case_passes": sum(row["gradient_passed"] for row in quality),
        "gradient_comparisons": len(gradient_rows),
        "gradient_comparison_passes": sum(row["passed"] for row in gradient_rows),
        "gradient_failures": failures,
        "skipped_cases": skipped,
    }
    (args.directory / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    # Mirror the plot aggregation: view means per job, median across jobs.
    grouped = {}
    for row in timings:
        key = (row["scope"], row["width"], row["height"], row["variant"])
        grouped.setdefault(key, {}).setdefault(row["job"], []).append(row)
    overview = []
    for scope, width, height in sorted({key[:3] for key in grouped}):
        row = {"scope": scope, "width": width, "height": height}
        for column in STAGES:
            medians = {}
            for variant in ("base", "pr"):
                jobs = grouped[(scope, width, height, variant)]
                medians[variant] = statistics.median(
                    statistics.mean(point[column] for point in points)
                    for points in jobs.values()
                )
                row[f"{variant}_{column}"] = medians[variant]
            row[f"speedup_{column}"] = medians["base"] / medians["pr"]
        overview.append(row)
    write_csv(args.directory / "overview.csv", overview)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
