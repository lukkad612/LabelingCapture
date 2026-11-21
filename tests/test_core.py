import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geometry import (
    build_homography,
    default_source_points,
    default_source_points_for_box,
    default_target_points,
    map_points,
)
from annotation import magic_wand_polygon
from modeling import predict_annotations
from storage import encode_yolo_labels, save_sample


class PerspectiveTests(unittest.TestCase):
    def test_default_source_quad_uses_selected_box_coordinates(self) -> None:
        box = [100, 50, 300, 450]
        source = default_source_points_for_box(box)
        np.testing.assert_allclose(
            source,
            [[100, 50], [300, 50], [300, 450], [100, 450]],
        )
        matrix = build_homography(source, default_target_points())
        np.testing.assert_allclose(
            map_points(source, matrix),
            default_target_points(),
            atol=1e-4,
        )

    def test_default_homography_maps_source_corners_to_target_corners(self) -> None:
        source = default_source_points(300, 200)
        target = default_target_points(50, 100)
        matrix = build_homography(source, target)

        np.testing.assert_allclose(map_points(source, matrix), target, atol=1e-4)

    def test_crossed_points_are_rejected(self) -> None:
        source = np.float32([[0, 0], [100, 100], [100, 0], [0, 100]])
        with self.assertRaises(ValueError):
            build_homography(source, default_target_points())

    def test_empty_point_mapping_has_yolo_shape(self) -> None:
        self.assertEqual(map_points(np.empty((0, 2)), np.eye(3)).shape, (0, 2))


class DatasetTests(unittest.TestCase):
    def test_pose_rows_have_consistent_keypoint_slots_and_visibility(self) -> None:
        annotations = [
            {"box": (10, 10, 50, 50), "class_id": 1, "keypoints": [(20, 30)]},
            {"box": (0, 0, 100, 100), "class_id": 0, "keypoints": []},
        ]
        rows = encode_yolo_labels(annotations, "pose", 100, 100).splitlines()

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0].split()), 8)
        self.assertEqual(len(rows[1].split()), 8)
        self.assertEqual(rows[0].split()[-1], "2")
        self.assertEqual(rows[1].split()[-3:], ["0", "0", "0"])

    def test_segment_rows_encode_normalized_polygon_coordinates(self) -> None:
        annotations = [
            {
                "box": [1, 1, 9, 9],
                "class_id": 2,
                "polygon": [(1, 1), (9, 1), (9, 9), (1, 9)],
            }
        ]
        row = encode_yolo_labels(annotations, "segment", 10, 10).split()
        self.assertEqual(row[0], "2")
        np.testing.assert_allclose(
            [float(value) for value in row[1:]],
            [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            atol=1e-8,
        )
        with self.assertRaises(ValueError):
            encode_yolo_labels(
                [{"box": [0, 0, 1, 1], "class_id": 0, "polygon": [(0, 0)]}],
                "segment",
                10,
                10,
            )

    def test_magic_wand_returns_seed_connected_region_boundary(self) -> None:
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[2:8, 3:10] = (120, 45, 200)
        image[9:11, 9:11] = (120, 45, 200)

        polygon = magic_wand_polygon(image, (4, 4), 0)
        points = np.asarray(polygon)

        self.assertGreaterEqual(len(polygon), 4)
        self.assertEqual(int(np.min(points[:, 0])), 3)
        self.assertEqual(int(np.max(points[:, 0])), 9)
        self.assertEqual(int(np.min(points[:, 1])), 2)
        self.assertEqual(int(np.max(points[:, 1])), 7)

    def test_prelabel_reports_missing_model_runtime(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaisesRegex(RuntimeError, "模型运行依赖未安装"):
                predict_annotations(image, Path("missing.pt"), ["object"], "detect")

    def test_sample_writes_image_label_and_required_metadata(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        source = default_source_points(30, 20)
        target = default_target_points()
        matrix = build_homography(source, target)
        annotations = [{"box": [1, 2, 10, 12], "class_id": 0, "keypoints": []}]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_id = save_sample(
                root,
                image,
                "pose",
                annotations,
                source,
                target,
                matrix,
                ["object"],
                {"annotation_index": 0, "box": [1, 2, 10, 12]},
            )

            self.assertTrue((root / "images" / f"{sample_id}.png").is_file())
            label = (root / "labels" / f"{sample_id}.txt").read_text(
                encoding="utf-8"
            ).split()
            self.assertEqual(label[0], "0")
            np.testing.assert_allclose(
                [float(value) for value in label[1:]],
                [5.5 / 30, 7 / 20, 9 / 30, 10 / 20],
                atol=1e-8,
            )
            metadata = json.loads(
                (root / "meta" / f"{sample_id}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["image"], {"width": 30, "height": 20, "format": "png"})
            self.assertEqual(metadata["task"], "pose")
            self.assertEqual(len(metadata["perspective"]["homography"]), 3)
            self.assertEqual(
                metadata["perspective"]["target"]["annotation_index"], 0
            )
            self.assertEqual(metadata["classes"][0]["name"], "object")


if __name__ == "__main__":
    unittest.main()
