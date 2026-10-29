"""Plot paired PR timing and correctness CSVs without running GPU workloads.

Timing rows: scope,width,height,camera_index,variant,job,forward_ms,
backward_ms,combined_ms,inference_ms. Quality rows: scope,width,height,
camera_index,rgb_psnr_db,rgb_exact,image_max_abs,alpha_max_abs,depth_max_abs,
grad_max_rel_l2,baseline_noise_max_rel_l2,gradient_passed,forward_passed.
Blank PSNR means positive infinity. Boolean fields accept true/false or 1/0.
Optional gradient rows: scope,width,height,camera_index,gradient,rel_l2,max_abs,
baseline_repeat_rel_l2,candidate_repeat_rel_l2,passed.
"""

import argparse
import csv
from pathlib import Path
import typing as t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCOPES = ("full", "raster")
STAGES = ("forward_ms", "backward_ms", "combined_ms")
STAGE_LABELS = ("Training forward", "Backward", "Forward + backward")
BASE_COLOR = "#2878B5"
PR_COLOR = "#E78524"
PSNR_CAP = 120.0
ZERO_FLOOR = 1e-16
GRAD_THRESHOLD = 1e-4
TIMING_COLUMNS = {
    "scope",
    "width",
    "height",
    "camera_index",
    "variant",
    "job",
    *STAGES,
    "inference_ms",
}
QUALITY_COLUMNS = {
    "scope",
    "width",
    "height",
    "camera_index",
    "rgb_psnr_db",
    "rgb_exact",
    "image_max_abs",
    "alpha_max_abs",
    "depth_max_abs",
    "grad_max_rel_l2",
    "baseline_noise_max_rel_l2",
    "gradient_passed",
    "forward_passed",
}
GRADIENT_COLUMNS = {
    "scope",
    "width",
    "height",
    "camera_index",
    "gradient",
    "rel_l2",
    "max_abs",
    "baseline_repeat_rel_l2",
    "candidate_repeat_rel_l2",
    "passed",
}
GRADIENT_GROUPS = {
    "full": (
        "grad.means",
        "grad.quats",
        "grad.log_scales",
        "grad.opacity_logits",
        "grad.sh_coefficients",
        "grad.means2d_abs",
    ),
    "raster": (
        "grad.means2d",
        "grad.conics",
        "grad.colors",
        "grad.opacities",
        "grad.means2d_abs",
    ),
}


def aggregate_gradients(rows: t.List[t.Dict[str, str]], view_sets: dict) -> dict:
    grouped = {}
    fields = ("rel_l2", "baseline_repeat_rel_l2", "candidate_repeat_rel_l2")
    for row in rows:
        size = resolution(row)
        scope, gradient = row["scope"], row["gradient"]
        if gradient not in GRADIENT_GROUPS[scope]:
            raise ValueError(f"Unknown gradient for {scope}: {gradient!r}")
        key = (scope, size, gradient)
        views = grouped.setdefault(key, {})
        camera = row["camera_index"]
        if camera in views:
            raise ValueError(f"Duplicate gradient row: {key}, camera {camera}")
        number(row["max_abs"], "max_abs")
        views[camera] = {
            **{field: number(row[field], field) for field in fields},
            "passed": boolean(row["passed"]),
        }
    expected = {
        (scope, size, gradient)
        for scope, size in view_sets
        for gradient in GRADIENT_GROUPS[scope]
    }
    if set(grouped) != expected:
        raise ValueError(
            "Gradient groups/resolutions must match the timing scopes/resolutions"
        )
    result = {}
    for (scope, size, gradient), views in grouped.items():
        if set(views) != view_sets[(scope, size)]:
            raise ValueError(
                f"Gradient and timing views differ: {scope}, {size}, {gradient}"
            )
        result[(scope, size, gradient)] = {
            **{field: max(item[field] for item in views.values()) for field in fields},
            "passed": all(item["passed"] for item in views.values()),
        }
    return result


def plot_gradients(
    scope: str, gradients: dict, output_dir: Path, base_label: str, pr_label: str
) -> None:
    dims = sorted(
        {key[1] for key in gradients if key[0] == scope},
        key=lambda size: (size[0] * size[1], size),
    )
    x = np.array([width * height / 1e6 for width, height in dims])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{'Full renderer parameters' if scope == 'full' else 'Raster attributes'}"
        " · per-field gradient agreement",
        fontsize=19,
        fontweight="bold",
    )
    for ax, gradient in zip(axes.flat, GRADIENT_GROUPS[scope]):
        checks = [gradients[(scope, size, gradient)] for size in dims]
        for field, label, color, style, marker in (
            ("rel_l2", "PR vs base", PR_COLOR, "-", "o"),
            ("baseline_repeat_rel_l2", "Base-repeat noise", "#479A70", "--", "s"),
            ("candidate_repeat_rel_l2", "PR-repeat noise", "#8856A7", ":", "^"),
        ):
            values = np.array([item[field] for item in checks])
            ax.plot(
                x,
                np.maximum(values, ZERO_FLOOR),
                color=color,
                linestyle=style,
                marker=marker,
                linewidth=2,
                label=label,
            )
        failures = np.array([not item["passed"] for item in checks])
        if failures.any():
            values = np.maximum([item["rel_l2"] for item in checks], ZERO_FLOOR)
            ax.scatter(
                x[failures],
                values[failures],
                color="red",
                marker="x",
                s=90,
                zorder=5,
                label="Field check failed",
            )
        ax.axhline(
            GRAD_THRESHOLD,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="Threshold: 1e−4",
        )
        ax.set(
            title=gradient,
            xlabel="Image size (megapixels)",
            ylabel="Worst-view relative L2 error",
            yscale="log",
        )
        ax.grid(True, alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
        if len(x) <= 10 and len(set(x)) == len(x):
            ax.set_xticks(
                x,
                [f"{mp:g}\n{w}×{h}" for mp, (w, h) in zip(x, dims)],
                rotation=35 if len(x) > 5 else 0,
                fontsize=8,
            )
    for ax in list(axes.flat)[len(GRADIENT_GROUPS[scope]) :]:
        ax.set_visible(False)
    fig.text(
        0.5,
        0.025,
        f"Base: {base_label} · PR: {pr_label}\n"
        "Each field/curve: maximum relative L2 across selected views; "
        "repeat-noise curves are separate within-variant comparisons.\n"
        "Display floor: 1e−16 only; zeros remain zero in the data, not measured 1e−16. "
        "MP = width × height / 1e6. Lines connect measured points only.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.95), h_pad=2.5, w_pad=2)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"gradients_{scope}.{extension}", dpi=300, facecolor="white"
        )
    plt.close(fig)


def read_csv(path: Path, required: t.Set[str]) -> t.List[t.Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError(f"Invalid boolean: {value!r}")
    return normalized in {"true", "1"}


def number(value: str, name: str, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def resolution(row: t.Dict[str, str]) -> t.Tuple[int, int]:
    if row["scope"] not in SCOPES:
        raise ValueError(f"Unknown scope: {row['scope']!r}")
    width, height = int(row["width"]), int(row["height"])
    if min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    return width, height


def aggregate_timings(rows: t.List[t.Dict[str, str]]) -> t.Tuple[dict, dict]:
    """Return median/min/max of job means, never a confidence interval."""
    grouped = {}
    for row in rows:
        dims = resolution(row)
        if row["variant"] not in {"base", "pr"}:
            raise ValueError(f"Unknown variant: {row['variant']!r}")
        key = (row["scope"], dims, row["variant"], row["job"])
        views = grouped.setdefault(key, {})
        camera = row["camera_index"]
        if camera in views:
            raise ValueError(f"Duplicate timing row: {key}, camera {camera}")
        views[camera] = [
            number(row[stage], stage, positive=True)
            for stage in (*STAGES, "inference_ms")
        ]
    by_resolution = {}
    view_sets = {}
    for (scope, dims, variant, job), views in grouped.items():
        key = (scope, dims)
        cameras = set(views)
        if key in view_sets and cameras != view_sets[key]:
            raise ValueError(f"Selected views differ across jobs/variants at {key}")
        view_sets[key] = cameras
        by_resolution.setdefault((scope, dims, variant), []).append(
            np.mean(list(views.values()), axis=0)
        )
    result = {}
    for key, means in by_resolution.items():
        values = np.asarray(means)
        result[key] = (
            np.median(values, axis=0),
            np.min(values, axis=0),
            np.max(values, axis=0),
        )
    return result, view_sets


def aggregate_quality(rows: t.List[t.Dict[str, str]]) -> dict:
    grouped = {}
    for row in rows:
        key = (row["scope"], resolution(row))
        views = grouped.setdefault(key, {})
        if row["camera_index"] in views:
            raise ValueError(
                f"Duplicate quality row: {key}, camera {row['camera_index']}"
            )
        psnr = float(row["rgb_psnr_db"]) if row["rgb_psnr_db"].strip() else np.inf
        if np.isnan(psnr) or psnr == -np.inf:
            raise ValueError("RGB PSNR must be finite or positive infinity")
        exact = boolean(row["rgb_exact"])
        if exact and not np.isposinf(psnr):
            raise ValueError("Exact RGB must have infinite (or blank) PSNR")
        for field in ("image_max_abs", "alpha_max_abs", "depth_max_abs"):
            number(row[field], field)
        views[row["camera_index"]] = {
            "psnr": psnr,
            "gradient": number(row["grad_max_rel_l2"], "grad_max_rel_l2"),
            "noise": number(
                row["baseline_noise_max_rel_l2"], "baseline_noise_max_rel_l2"
            ),
            "forward_passed": boolean(row["forward_passed"]),
            "gradient_passed": boolean(row["gradient_passed"]),
        }
    result = {}
    for key, views in grouped.items():
        values = list(views.values())
        result[key] = {
            "psnr": min(item["psnr"] for item in values),
            "gradient": max(item["gradient"] for item in values),
            "noise": max(item["noise"] for item in values),
            "forward_passed": all(item["forward_passed"] for item in values),
            "gradient_passed": all(item["gradient_passed"] for item in values),
            "cameras": set(views),
        }
    return result


def plot_scope(
    scope: str,
    timings: dict,
    quality: dict,
    view_sets: dict,
    output_dir: Path,
    base_label: str,
    pr_label: str,
) -> None:
    dims = sorted(
        {key[1] for key in timings if key[0] == scope},
        key=lambda size: (size[0] * size[1], size),
    )
    if not dims:
        raise ValueError(f"No timing rows for scope {scope!r}")
    for size in dims:
        for variant in ("base", "pr"):
            if (scope, size, variant) not in timings:
                raise ValueError(f"Missing {variant} timings for {scope}, {size}")
        key = (scope, size)
        if key not in quality or quality[key]["cameras"] != view_sets[key]:
            raise ValueError(f"Quality and timing views must match for {key}")
    if {key[1] for key in quality if key[0] == scope} != set(dims):
        raise ValueError(f"Quality and timing resolutions differ for {scope}")

    x = np.array([width * height / 1e6 for width, height in dims])
    summaries = {
        variant: np.array([timings[(scope, size, variant)] for size in dims])
        for variant in ("base", "pr")
    }
    checks = [quality[(scope, size)] for size in dims]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{'Full renderer' if scope == 'full' else 'Rasterization only'} · megapixel scaling",
        fontsize=19,
        fontweight="bold",
    )
    for stage, title in enumerate(STAGE_LABELS):
        ax = axes[0, stage]
        for variant, color, label in (
            ("base", BASE_COLOR, base_label),
            ("pr", PR_COLOR, pr_label),
        ):
            median, low, high = summaries[variant][:, :, stage].T
            ax.errorbar(
                x,
                median,
                yerr=np.array([median - low, high - median]),
                color=color,
                label=label,
                marker="o",
                capsize=4,
                linewidth=2,
            )
        ax.set(title=title, ylabel="Latency (ms)", ylim=(0, None))
        ax.legend(fontsize=9)

    ax = axes[1, 0]
    for stage, (label, color, marker) in enumerate(
        zip(STAGE_LABELS, (BASE_COLOR, PR_COLOR, "#479A70"), ("o", "s", "^"))
    ):
        ratio = summaries["base"][:, 0, stage] / summaries["pr"][:, 0, stage]
        ax.plot(x, ratio, marker=marker, color=color, label=label, linewidth=2)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=1, label="1× reference")
    ax.set(title="Render component speedup", ylabel="Base / PR median latency (×)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    psnr = np.array([item["psnr"] for item in checks])
    displayed_psnr = np.where(np.isposinf(psnr), PSNR_CAP, psnr)
    ax.plot(x, displayed_psnr, "o-", color=PR_COLOR, label="Worst view RGB PSNR")
    for position in np.flatnonzero(np.isposinf(psnr)):
        ax.annotate(
            "∞",
            (x[position], PSNR_CAP),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )
    failures = np.array([not item["forward_passed"] for item in checks])
    if failures.any():
        ax.scatter(
            x[failures],
            displayed_psnr[failures],
            color="red",
            marker="x",
            s=90,
            zorder=5,
            label="Forward check failed",
        )
    ax.set(title="RGB agreement across views", ylabel="PSNR (dB; higher is better)")
    ax.margins(y=0.2)
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    for field, label, color, marker in (
        ("gradient", "PR vs base: worst gradient", PR_COLOR, "o"),
        ("noise", "Base-repeat noise: worst gradient", BASE_COLOR, "s"),
    ):
        values = np.array([item[field] for item in checks])
        ax.plot(
            x,
            np.maximum(values, ZERO_FLOOR),
            marker=marker,
            color=color,
            label=label,
            linewidth=2,
        )
    failures = np.array([not item["gradient_passed"] for item in checks])
    if failures.any():
        values = np.maximum([item["gradient"] for item in checks], ZERO_FLOOR)
        ax.scatter(
            x[failures],
            values[failures],
            color="red",
            marker="x",
            s=90,
            zorder=5,
            label="Gradient check failed",
        )
    ax.axhline(
        GRAD_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label="Threshold: 1e−4",
    )
    ax.set(
        title="Gradient agreement across views",
        ylabel="Worst relative L2 error",
        yscale="log",
    )
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Display floor: 1e−16 (zeros are not measured as 1e−16)",
        transform=ax.transAxes,
        fontsize=7,
    )

    for ax in axes.flat:
        ax.set_xlabel("Image size (megapixels)")
        ax.grid(True, alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if len(x) <= 10 and len(set(x)) == len(x):
            ax.set_xticks(
                x,
                [f"{mp:g}\n{w}×{h}" for mp, (w, h) in zip(x, dims)],
                rotation=35 if len(x) > 5 else 0,
                fontsize=8,
            )
    fig.text(
        0.5,
        0.025,
        "Timing: mean across selected views per job → median across job means; "
        "whiskers = job-mean min/max (not CI). Speedup = ratio of medians.\n"
        "Forward uses training autograd; no inference timings shown. "
        "Quality: minimum PSNR / maximum gradient error and repeat noise across views.\n"
        "MP = width × height / 1e6. Infinite PSNR is displayed at 120 dB only "
        "(∞ label); finite PSNR is not clipped. Lines connect measured points only.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.95), h_pad=2.5, w_pad=2)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{scope}_megapixels.{extension}", dpi=300, facecolor="white"
        )
    plt.close(fig)


def plot_inference(timings: dict, output_dir: Path, base_label: str, pr_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Forward-only inference · no autograd", fontsize=18, fontweight="bold")
    for ax, scope in zip(axes, SCOPES):
        sizes = sorted(
            {key[1] for key in timings if key[0] == scope},
            key=lambda size: size[0] * size[1],
        )
        x = np.array([width * height / 1e6 for width, height in sizes])
        for variant, color, label in (
            ("base", BASE_COLOR, base_label),
            ("pr", PR_COLOR, pr_label),
        ):
            med, low, high = np.array(
                [timings[(scope, size, variant)] for size in sizes]
            )[:, :, 3].T
            ax.errorbar(
                x,
                med,
                yerr=[med - low, high - med],
                marker="o",
                color=color,
                label=label,
                capsize=4,
                linewidth=2,
            )
        ax.set(
            title="Full renderer" if scope == "full" else "Rasterization only",
            xlabel="Image size (megapixels)",
            ylabel="Forward-only latency (ms)",
            ylim=(0, None),
        )
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x, [f"{value:g}" for value in x], rotation=35, fontsize=8)
    fig.text(
        0.5,
        0.025,
        "No-grad forward is measured separately; it is not added to training backward.\n"
        "Mean of selected views per job, median over job means; whiskers=min/max (not CI).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"inference_only.{extension}", dpi=300, facecolor="white"
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings-csv", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--gradients-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-label", default="PR base ffe2db3")
    parser.add_argument("--pr-label", default="PR 3c69277")
    args = parser.parse_args()
    try:
        timings, views = aggregate_timings(read_csv(args.timings_csv, TIMING_COLUMNS))
        quality = aggregate_quality(read_csv(args.quality_csv, QUALITY_COLUMNS))
        gradients = (
            aggregate_gradients(read_csv(args.gradients_csv, GRADIENT_COLUMNS), views)
            if args.gradients_csv is not None
            else None
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(
            {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
        ):
            plot_inference(timings, args.output_dir, args.base_label, args.pr_label)
            for scope in SCOPES:
                plot_scope(
                    scope,
                    timings,
                    quality,
                    views,
                    args.output_dir,
                    args.base_label,
                    args.pr_label,
                )
                if gradients is not None:
                    plot_gradients(
                        scope,
                        gradients,
                        args.output_dir,
                        args.base_label,
                        args.pr_label,
                    )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
