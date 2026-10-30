"""Convert a PINHOLE COLMAP text sparse model to PR9 camera JSON.

Images are ordered by image ID. The second (observations) line is consumed even
when empty. This is an input adapter, not a renderer or benchmark setting change.
"""

import argparse
import json
from pathlib import Path
import typing as t

import numpy as np


def convert(sparse: Path) -> t.List[t.Dict[str, t.Any]]:
    cameras = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8 or fields[1] != "PINHOLE":
            raise ValueError("This adapter requires PINHOLE COLMAP text cameras")
        cameras[int(fields[0])] = dict(
            width=int(fields[2]),
            height=int(fields[3]),
            **dict(zip(("fx", "fy", "cx", "cy"), map(float, fields[4:]))),
        )
    result = []
    with (sparse / "images.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Malformed COLMAP image header")
            q = np.asarray(list(map(float, fields[1:5])), dtype=np.float64)
            norm = np.linalg.norm(q)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Invalid COLMAP quaternion")
            w, x, y, z = q / norm
            rotation = np.array(
                [
                    [
                        1 - 2 * y * y - 2 * z * z,
                        2 * x * y - 2 * w * z,
                        2 * x * z + 2 * w * y,
                    ],
                    [
                        2 * x * y + 2 * w * z,
                        1 - 2 * x * x - 2 * z * z,
                        2 * y * z - 2 * w * x,
                    ],
                    [
                        2 * x * z - 2 * w * y,
                        2 * y * z + 2 * w * x,
                        1 - 2 * x * x - 2 * y * y,
                    ],
                ]
            )
            translation = np.asarray(list(map(float, fields[5:8])))
            result.append(
                dict(
                    id=int(fields[0]),
                    img_name=fields[9],
                    **cameras[int(fields[8])],
                    rotation=rotation.T.tolist(),
                    position=(-rotation.T @ translation).tolist(),
                )
            )
            if next(stream, None) is None:
                raise ValueError("Missing COLMAP image observations line")
    return sorted(result, key=lambda camera: camera["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(convert(args.sparse), indent=2) + "\n")


if __name__ == "__main__":
    main()
