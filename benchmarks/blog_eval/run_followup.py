"""Serial, alternating PR9-protocol jobs for a baseline and candidate binary.

References must live outside the repository. The existing megapixel worker owns
all timing, workload identity and parity rules; this driver does not change them.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import typing as t


def telemetry() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,clocks.sm,power.draw",
            "--format=csv",
        ],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--camera-indices", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--resolutions", nargs="+")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: t.Dict[str, t.Any] = {"jobs": [], "parity_failures": []}
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    for job in range(1, args.jobs + 1):
        order = ("baseline", "candidate") if job % 2 else ("candidate", "baseline")
        for variant in order:
            prefix = "base" if variant == "baseline" else "pr"
            output = args.output_dir / f"{prefix}-job{job}.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("benchmark_megapixels.py")),
                "--extension",
                str(getattr(args, variant).resolve()),
                "--ply",
                str(args.ply.resolve()),
                "--cameras",
                str(args.cameras.resolve()),
                "--camera-indices",
                *args.camera_indices,
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--max-points",
                "0",
                "--scope",
                "both",
                "--label",
                variant,
                "--output",
                str(output.resolve()),
            ]
            if args.resolutions:
                command.extend(["--resolutions", *args.resolutions])
            if job == 1:
                action = "save" if variant == "baseline" else "compare"
                command.extend(
                    [f"--{action}-reference-dir", str(args.reference_dir.resolve())]
                )
            else:
                command.append("--skip-validation")
            entry = {
                "job": job,
                "variant": variant,
                "command": command,
                "telemetry_before": telemetry(),
            }
            start = time.monotonic()
            with output.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            entry.update(
                exit_code=result.returncode,
                elapsed_seconds=time.monotonic() - start,
                telemetry_after=telemetry(),
            )
            manifest["jobs"].append(entry)
            (args.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            if result.returncode:
                raise RuntimeError(
                    f"{variant} job {job} failed; see {output.with_suffix('.log')}"
                )
            report = json.loads(output.read_text())
            for record in report["records"]:
                if record["status"] != "ok":
                    raise RuntimeError(
                        f"Invalid record in {output}: {record['camera_index']}, {record['resolution']}, {record['scope']}"
                    )
                if record.get("alignment_passed") is False:
                    manifest["parity_failures"].append(
                        {
                            "output": str(output),
                            "camera_index": record["camera_index"],
                            "resolution": record["resolution"],
                            "scope": record["scope"],
                        }
                    )
                for checks in record["finite_checks"].values():
                    if not all(checks.values()):
                        raise RuntimeError(f"Nonfinite values in {output}")
            print(
                f"{variant} job {job}: {len(report['records'])} records; "
                f"{len(manifest['parity_failures'])} accumulated parity failures",
                flush=True,
            )
            (args.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    if manifest["parity_failures"]:
        raise RuntimeError(
            "Parity failures remain; see manifest.json (timings retained)"
        )


if __name__ == "__main__":
    main()
