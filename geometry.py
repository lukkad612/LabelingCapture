from __future__ import annotations

import cv2
import numpy as np


def default_source_points(width: int, height: int) -> np.ndarray:
    inset_x = min(10.0, max(0.0, (width - 1) * 0.05))
    inset_y = min(10.0, max(0.0, (height - 1) * 0.05))
    right = max(inset_x + 1.0, width - 1.0 - inset_x)
    bottom = max(inset_y + 1.0, height - 1.0 - inset_y)
    return np.float32(
        [[inset_x, inset_y], [right, inset_y], [right, bottom], [inset_x, bottom]]
    )


def default_source_points_for_box(
    box: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    left, top, right, bottom = map(float, box)
    return np.float32(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]
    )


def default_target_points(width: int = 50, height: int = 100) -> np.ndarray:
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]])


def validate_quad(points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("四点必须是四组有效的 x、y 坐标")

    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(np.abs(crosses) < 1e-6) or not (
        np.all(crosses > 0) or np.all(crosses < 0)
    ):
        raise ValueError("四点必须按左上、右上、右下、左下顺序组成凸四边形")

    area = 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if area < 1e-4:
        raise ValueError("四点围成的区域过小")


def build_homography(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    validate_quad(source_points)
    validate_quad(target_points)
    matrix = cv2.getPerspectiveTransform(
        np.float32(source_points), np.float32(target_points)
    )
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("透视变换矩阵无效")
    return matrix


def map_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def rotated_box_points(box, angle: float) -> np.ndarray:
    left, top, right, bottom = box
    center = np.array([(left + right) / 2, (top + bottom) / 2])
    corners = np.array([(left, top), (right, top), (right, bottom), (left, bottom)])
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    return (corners - center) @ rotation + center


def resize_rotated_box(
    box: list[float],
    angle: float,
    corner: int,
    dx: float,
    dy: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    corners = rotated_box_points(box, angle)
    fixed = corners[(corner + 2) % 4]
    initial_size = np.array([box[2] - box[0], box[3] - box[1]])
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    horizontal = [cosine, sine]
    vertical = [-sine, cosine]
    signs = np.array([1 if corner in (1, 2) else -1, 1 if corner in (2, 3) else -1])
    axes = np.array([horizontal, vertical]) * signs[:, None]
    size = np.maximum(np.where(initial_size > 0, np.minimum(2.0, initial_size), 2.0), initial_size + axes @ [dx, dy])

    # Limit the drag to the first image edge any corner would cross.
    # An already out-of-bounds box can still be resized back into the image.
    limit = np.array([image_width, image_height])
    if np.all(corners >= -1e-8) and np.all(corners <= limit + 1e-8):
        shifts = (size - initial_size)[:, None] * axes
        scale = 1.0
        for origin, shift in zip((fixed + initial_size[0] * axes[0],
                                  fixed + initial_size[1] * axes[1],
                                  corners[corner]), (*shifts, shifts.sum(axis=0))):
            for coordinate, motion, bound in zip(origin, shift, limit):
                if motion > 0:
                    scale = min(scale, (bound - coordinate) / motion)
                elif motion < 0:
                    scale = min(scale, -coordinate / motion)
        size = initial_size + max(0.0, scale) * (size - initial_size)

    center = fixed + size @ axes / 2
    return np.concatenate((center - size / 2, center + size / 2)).tolist()


def map_box(box: tuple[float, float, float, float], matrix: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return map_points(corners, matrix)
