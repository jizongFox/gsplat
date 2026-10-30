"""Convert undistorted COLMAP text cameras/images to world-frame 3DGS JSON."""

import argparse
import json
from pathlib import Path

import torch

from renderer_benchmark import quaternion_columns


def convert(cameras_path, images_path):
    cameras = {}
    for line in cameras_path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        cid, model, width, height = fields[:4]
        params = list(map(float, fields[4:]))
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(f"Camera {cid}: only undistorted PINHOLE/SIMPLE_PINHOLE supported")
        cameras[int(cid)] = dict(width=int(width), height=int(height), fx=fx, fy=fy, cx=cx, cy=cy)
    result = []
    lines = iter(images_path.read_text().splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        image_id = int(fields[0])
        q = torch.tensor([list(map(float, fields[1:5]))], dtype=torch.float64)
        rotation = quaternion_columns(q)[0].T  # COLMAP wxyz rotation is world-to-camera.
        translation = torch.tensor(list(map(float, fields[5:8])), dtype=torch.float64)
        result.append({**cameras[int(fields[8])], "id": image_id, "img_name": " ".join(fields[9:]),
                       "rotation": rotation.T.tolist(), "position": (-rotation.T @ translation).tolist()})
        next(lines)  # Consume POINTS2D line even when empty.
    return sorted(result, key=lambda camera: camera["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-txt", type=Path, required=True)
    parser.add_argument("--images-txt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converted = convert(args.cameras_txt, args.images_txt)
    with args.output.open("x") as stream:
        json.dump(converted, stream, indent=2, allow_nan=False)
        stream.write("\n")
