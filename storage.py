from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np

from geometry import rotated_box_points
from naming import numbered_output


def _normalized_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    )


def encode_yolo_labels(
    annotations: list[dict[str, Any]],
    task: str,
    width: int,
    height: int,
) -> str:
    lines: list[str] = []
    if task == "obb":
        for annotation in annotations:
            points = rotated_box_points(annotation["box"], annotation.get("angle", 0))
            raw = points / [width, height]
            if not np.isfinite(raw).all() or (raw < -1e-8).any() or (raw > 1 + 1e-8).any():
                raise ValueError("旋转框的四个角必须位于图片内。")
            coordinates = " ".join(f"{value:.8f}" for value in raw.clip(0, 1).flat)
            lines.append(f"{int(annotation['class_id'])} {coordinates}")
        return "\n".join(lines) + ("\n" if lines else "")
    if task == "segment":
        for annotation in annotations:
            polygon = annotation.get("polygon")
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                raise ValueError("实例分割标注必须包含至少 3 个多边形点")
            values = [str(int(annotation["class_id"]))]
            for point in polygon:
                try:
                    coordinates = np.asarray(point, dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "分割多边形中的每个点必须包含 x、y 坐标"
                    ) from error
                if coordinates.shape != (2,):
                    raise ValueError("分割多边形中的每个点必须包含 x、y 坐标")
                x, y = coordinates.tolist()
                if not np.isfinite(coordinates).all():
                    raise ValueError("分割多边形坐标必须是有限数值")
                values.extend(
                    (
                        f"{min(max(x / width, 0.0), 1.0):.8f}",
                        f"{min(max(y / height, 0.0), 1.0):.8f}",
                    )
                )
            lines.append(" ".join(values))
        return "\n".join(lines) + ("\n" if lines else "")

    keypoint_count = max(
        (len(annotation.get("keypoints", [])) for annotation in annotations),
        default=0,
    )
    for annotation in annotations:
        box = _normalized_box(tuple(annotation["box"]), width, height)
        values = [str(int(annotation["class_id"])), *(f"{value:.8f}" for value in box)]
        if task == "pose":
            keypoints = annotation.get("keypoints", [])
            visibility = annotation.get("keypoint_visibility", [])
            for index in range(keypoint_count):
                point = keypoints[index] if index < len(keypoints) else None
                if point is None:
                    values.extend(("0", "0", "0"))
                    continue
                x, y = point
                values.extend(
                    (
                        f"{min(max(x / width, 0.0), 1.0):.8f}",
                        f"{min(max(y / height, 0.0), 1.0):.8f}",
                        "1" if index < len(visibility) and visibility[index] == 1 else "2",
                    )
                )
        lines.append(" ".join(values))
    return "\n".join(lines) + ("\n" if lines else "")


def save_source_label(path: Path, expected: bytes | None, content: str) -> bytes:
    payload = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None
        if current != expected:
            raise ValueError("源标签已在其他程序中修改，请重新导入后再保存。")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def save_sample(
    dataset_dir: Path,
    image_rgb: np.ndarray,
    task: str,
    annotations: list[dict[str, Any]],
    source_points: np.ndarray,
    target_points: np.ndarray,
    homography: np.ndarray,
    classes: list[str],
    perspective_target: dict[str, Any] | None = None,
) -> str:
    height, width = image_rgb.shape[:2]
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    with numbered_output(dataset_dir, "sample_", ("images", "labels", "meta")) as sample_id:
        paths = {
            "images": dataset_dir / "images" / f"{sample_id}.png",
            "labels": dataset_dir / "labels" / f"{sample_id}.txt",
            "meta": dataset_dir / "meta" / f"{sample_id}.json",
        }
        image_ok, encoded_image = cv2.imencode(
            ".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        )
        if not image_ok:
            raise OSError("无法编码截图图像")

        meta = {
            "timestamp": timestamp,
            "task": task,
            "image": {"width": width, "height": height, "format": "png"},
            "perspective": {
                "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                "source_points": np.asarray(source_points, dtype=float).tolist(),
                "target_points": np.asarray(target_points, dtype=float).tolist(),
                "homography": np.asarray(homography, dtype=float).tolist(),
                "target": perspective_target,
            },
            "classes": [{"id": index, "name": name} for index, name in enumerate(classes)],
            "annotations": annotations,
        }
        payloads = {
            paths["labels"]: encode_yolo_labels(annotations, task, width, height).encode(
                "utf-8"
            ),
            paths["meta"]: (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        }
        written: list[Path] = []
        temporary: list[Path] = []
        try:
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            image_tmp = paths["images"].with_suffix(".png.tmp")
            temporary.append(image_tmp)
            image_tmp.write_bytes(encoded_image.tobytes())
            for target, payload in payloads.items():
                tmp = target.with_suffix(target.suffix + ".tmp")
                temporary.append(tmp)
                tmp.write_bytes(payload)
            for tmp, target in zip(temporary, paths.values()):
                os.replace(tmp, target)
                written.append(target)
        except OSError:
            for path in temporary:
                path.unlink(missing_ok=True)
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return sample_id
