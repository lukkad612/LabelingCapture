import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from annotation import Annotation, AnnotationCanvas, AnnotationDialog, _closed_edges
from capture_dialog import CameraDialog
from geometry import build_homography, default_source_points, map_points
from main import (
    AppController,
    DEFAULT_SETTINGS,
    _ensure_standard_streams,
    _validate_settings,
)
from settings_dialog import SettingsDialog


class _FakeListener:
    def __init__(self, **_callbacks) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        pass


class DialogLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def tearDown(self) -> None:
        for widget in QApplication.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_saving_or_cancelling_editor_releases_capture_state(self) -> None:
        settings = {
            "hotkey": "<ctrl>+<alt>+a",
            "dataset_dir": "dataset",
            "continuous_capture": False,
            "classes": ["object"],
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "main.keyboard.Listener", _FakeListener
        ):
            controller = AppController(self.app, Path(temporary), settings)
            controller.desktop_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            controller.desktop_origin = (0, 0)
            controller._selection_made(QRect(0, 0, 32, 24))

            first_editor = controller.editor
            self.assertIsNotNone(first_editor)
            self.app.processEvents()
            self.assertTrue(first_editor.preview_magnifier.isVisible())
            first_editor.reject()
            self.assertFalse(first_editor.preview_magnifier.isVisible())
            self.assertIsNone(controller.editor)

            controller.desktop_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            controller._selection_made(QRect(0, 0, 32, 24))
            second_editor = controller.editor
            self.assertIsNotNone(second_editor)
            second_editor.mark_saved()
            self.assertFalse(second_editor.preview_magnifier.isVisible())
            self.assertIsNone(controller.editor)
            controller.quit()

    def test_bundled_model_path_resolves_from_pyinstaller_internal_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "main.keyboard.Listener", _FakeListener
        ):
            app_dir = Path(temporary) / "app"
            bundle_dir = app_dir / "_internal"
            bundled_model = bundle_dir / "models" / "detect" / "YOLO11n.pt"
            bundled_model.parent.mkdir(parents=True)
            bundled_model.touch()
            with patch.object(sys, "_MEIPASS", str(bundle_dir), create=True):
                controller = AppController(self.app, app_dir, DEFAULT_SETTINGS.copy())
                self.assertEqual(controller._resolved_model_path(), bundled_model)
                controller.listener.stop()
                controller.tray.hide()

    def test_ctrl_wheel_zoom_preserves_cursor_image_position(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas = AnnotationCanvas(image, [], default_source_points(100, 100))
        canvas.resize(400, 300)
        point_before = canvas._widget_to_image(QPointF(200, 150))
        event = QWheelEvent(
            QPointF(200, 150),
            QPointF(200, 150),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

        QApplication.sendEvent(canvas, event)

        self.assertGreater(canvas._zoom, 1.0)
        point_after = canvas._widget_to_image(QPointF(200, 150))
        np.testing.assert_allclose(point_after, point_before, atol=1e-5)

    def test_wheel_pans_vertically_and_shift_wheel_pans_horizontally(self) -> None:
        canvas = AnnotationCanvas(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [],
            default_source_points(100, 100),
        )
        canvas.resize(400, 300)
        canvas._zoom = 2.0
        vertical_event = QWheelEvent(
            QPointF(200, 150),
            QPointF(200, 150),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(canvas, vertical_event)
        self.assertGreater(canvas._pan.y(), 0)
        old_y = canvas._pan.y()

        horizontal_event = QWheelEvent(
            QPointF(200, 150),
            QPointF(200, 150),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ShiftModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(canvas, horizontal_event)
        self.assertLess(canvas._pan.x(), 0)
        self.assertEqual(canvas._pan.y(), old_y)

    def test_box_creation_uses_two_diagonal_clicks(self) -> None:
        canvas = AnnotationCanvas(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [],
            default_source_points(100, 100),
        )
        canvas.resize(400, 300)

        def click(point: QPointF) -> None:
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                point,
                point,
                point,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPointingDevice.primaryPointingDevice(),
            )
            QApplication.sendEvent(canvas, event)
            release = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                point,
                point,
                point,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                QPointingDevice.primaryPointingDevice(),
            )
            QApplication.sendEvent(canvas, release)

        click(canvas._image_to_widget(10, 10))
        self.assertEqual(canvas.annotations, [])
        self.assertIsNotNone(canvas._draft_box)
        click(canvas._image_to_widget(80, 90))
        self.assertEqual(len(canvas.annotations), 1)
        np.testing.assert_allclose(canvas.annotations[0].box, [10, 10, 80, 90])

    def test_right_click_confirms_preview_box(self) -> None:
        canvas = AnnotationCanvas(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [],
            default_source_points(100, 100),
        )
        canvas.resize(400, 300)
        start = canvas._image_to_widget(10, 10)
        end = canvas._image_to_widget(80, 90)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start,
            start,
            start,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPointingDevice.primaryPointingDevice(),
        )
        QApplication.sendEvent(canvas, press)
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            end,
            end,
            end,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            QPointingDevice.primaryPointingDevice(),
        )
        QApplication.sendEvent(canvas, move)
        confirm = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            end,
            end,
            end,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            QPointingDevice.primaryPointingDevice(),
        )

        QApplication.sendEvent(canvas, confirm)

        self.assertEqual(len(canvas.annotations), 1)
        np.testing.assert_allclose(canvas.annotations[0].box, [10, 10, 80, 90])

    def test_delete_removes_selected_box_even_when_a_keypoint_is_selected(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotations = [
            Annotation([10, 10, 50, 50], keypoints=[(20, 20)]),
            Annotation([55, 55, 90, 90]),
        ]
        canvas = AnnotationCanvas(image, annotations, default_source_points(100, 100))
        canvas.selected_box = 0
        canvas.selected_keypoint = (0, 0)

        canvas.delete_selected()

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].box, [55, 55, 90, 90])
        self.assertEqual(canvas.selected_box, -1)
        self.assertIsNone(canvas.selected_keypoint)

    def test_preview_magnifier_follows_perspective_transformed_cursor(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        dialog = AnnotationDialog(
            image,
            ["object"],
            magnifier_zoom=4.0,
            crosshair_color=(12, 34, 56),
            box_color=(78, 90, 123),
            initial_task="pose",
        )
        dialog.resize(1380, 850)
        dialog.annotations.append(Annotation([20, 20, 80, 80]))
        dialog.canvas.selected_box = 0
        dialog._refresh_preview()
        self.assertEqual(dialog.canvas.crosshair_color.getRgb()[:3], (12, 34, 56))
        self.assertEqual(dialog.canvas.box_color.getRgb()[:3], (78, 90, 123))
        dialog.show()
        self.app.processEvents()

        cursor = (45.0, 55.0)
        expected = map_points(
            np.float32([cursor]),
            build_homography(dialog.source_points, dialog.target_points),
        )[0]
        dialog._pointer_position_changed(cursor)

        self.assertTrue(dialog.preview_magnifier.isVisible())
        self.assertEqual(dialog._magnifier_image.pixelColor(0, 0).getRgb()[:3], (0, 0, 0))
        self.assertAlmostEqual(
            dialog.preview_magnifier._source_rect.center().x(),
            float(expected[0]),
            places=4,
        )
        self.assertAlmostEqual(
            dialog.preview_magnifier._source_rect.center().y(),
            float(expected[1]),
            places=4,
        )
        self.assertTrue(dialog.isAncestorOf(dialog.preview_magnifier))
        dialog.reject()
        self.assertFalse(dialog.preview_magnifier.isVisible())

    def test_non_pose_magnifier_shows_image_and_keeps_last_position(self) -> None:
        dialog = AnnotationDialog(
            np.zeros((100, 100, 3), dtype=np.uint8), ["object"]
        )
        dialog.resize(1380, 850)
        dialog.show()
        self.app.processEvents()
        dialog._pointer_position_changed((42.0, 54.0))
        previous_rect = dialog.preview_magnifier._source_rect

        self.assertFalse(previous_rect.isEmpty())
        self.assertEqual(dialog._magnifier_image.width(), 100)
        dialog._pointer_position_changed(None)

        self.assertEqual(dialog.preview_magnifier._source_rect, previous_rect)
        dialog.close()

    def test_pose_keypoints_connect_in_order_and_close_with_thin_lines(self) -> None:
        points = [(30.0, 30.0), (90.0, 30.0), (90.0, 90.0), (30.0, 90.0)]
        self.assertEqual(
            _closed_edges(points),
            [
                (points[0], points[1]),
                (points[1], points[2]),
                (points[2], points[3]),
                (points[3], points[0]),
            ],
        )
        self.assertEqual(_closed_edges(points[:1]), [])

        dialog = AnnotationDialog(
            np.zeros((120, 120, 3), dtype=np.uint8), ["object"], initial_task="pose"
        )
        dialog.annotations.append(
            Annotation([10, 10, 110, 110], keypoints=points)
        )
        dialog.canvas.selected_box = 0
        dialog._refresh_preview()

        matrix = build_homography(dialog.source_points, dialog.target_points)
        mapped = map_points(np.float32(points), matrix)
        closure_midpoint = np.mean([mapped[3], mapped[0]], axis=0)
        pixel = dialog._preview_image.pixelColor(
            int(round(float(closure_midpoint[0]))),
            int(round(float(closure_midpoint[1]))),
        )
        self.assertGreater(pixel.blue(), pixel.red())
        self.assertGreater(pixel.green(), pixel.red())
        dialog.close()

    def test_rgb_colors_validate_and_are_exposed_in_settings(self) -> None:
        legacy_settings = {
            key: value
            for key, value in DEFAULT_SETTINGS.items()
            if key not in ("crosshair_color", "box_color")
        }
        validated = _validate_settings(legacy_settings)
        self.assertEqual(validated["crosshair_color"], [230, 235, 240])
        self.assertEqual(validated["box_color"], [50, 220, 110])
        with self.assertRaises(ValueError):
            _validate_settings({**DEFAULT_SETTINGS, "box_color": [0, 128, 256]})

        dialog = SettingsDialog(DEFAULT_SETTINGS, Path.cwd())
        dialog.crosshair_color[0].setValue(12)
        dialog.box_color[2].setValue(255)
        values = dialog.values()
        self.assertEqual(values["crosshair_color"], [12, 235, 240])
        self.assertEqual(values["box_color"], [50, 220, 255])
        self.assertEqual(values["max_models"], 5)
        self.assertEqual(values["magic_wand_tolerance"], 20)
        self.assertEqual(
            values["model_paths"],
            {
                "detect": "models/detect/YOLO11n.pt",
                "pose": "models/pose/yolo11n-pose.pt",
                "segment": "models/segment/yolo11n-seg.pt",
            },
        )
        dialog.close()

    def test_perspective_preview_drag_updates_matching_target_point(self) -> None:
        dialog = AnnotationDialog(
            np.zeros((100, 100, 3), dtype=np.uint8), ["object"], initial_task="pose"
        )
        dialog.annotations.append(Annotation([10, 10, 90, 90]))
        dialog.canvas.selected_box = 0
        dialog._refresh_preview()

        dialog._target_preview_point_dragged(1, 24.5, 37.5)

        np.testing.assert_allclose(dialog.target_points[1], [24.5, 37.5])
        self.assertEqual(dialog.target_controls[1][0].value(), 24.5)
        self.assertEqual(dialog.target_controls[1][1].value(), 37.5)
        dialog.close()

    def test_invalid_default_perspective_points_are_rejected(self) -> None:
        crossed = [[0, 0], [1, 1], [1, 0], [0, 1]]
        with self.assertRaises(ValueError):
            _validate_settings(
                {**DEFAULT_SETTINGS, "perspective_source_points": crossed}
            )

    def test_legacy_single_model_setting_migrates_to_detect_model(self) -> None:
        settings = DEFAULT_SETTINGS.copy()
        settings.pop("model_paths")
        settings.pop("last_task")
        settings["model_path"] = "custom.pt"

        validated = _validate_settings(settings)

        self.assertEqual(validated["model_paths"]["detect"], "custom.pt")
        self.assertEqual(validated["model_paths"]["pose"], "models/pose/yolo11n-pose.pt")
        self.assertEqual(validated["last_task"], "detect")

    def test_dragging_box_edge_moves_box_instead_of_resizing(self) -> None:
        canvas = AnnotationCanvas(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [Annotation([10, 10, 90, 90])],
            default_source_points(100, 100),
        )
        canvas.resize(420, 320)
        canvas.selected_box = 0
        edge_position = canvas._image_to_widget(10, 50)

        self.assertIsNone(canvas._hit_box_handle(edge_position))
        self.assertEqual(canvas._hit_box((10, 50)), 0)

    def test_dragging_box_past_canvas_edge_clamps_to_image_boundary(self) -> None:
        canvas = AnnotationCanvas(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [Annotation([10, 10, 90, 90])],
            default_source_points(100, 100),
        )
        canvas.resize(420, 320)
        canvas.selected_box = 0
        canvas._drag_kind = "move"
        canvas._drag_start = QPointF(50, 50)
        canvas._initial_box = canvas.annotations[0].box.copy()
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(-50, 160),
            QPointF(-50, 160),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        QApplication.sendEvent(canvas, event)

        self.assertEqual(canvas.annotations[0].box[0], 0)
        self.assertEqual(canvas.annotations[0].box[2], 80)

    def test_switching_task_clears_annotations_after_confirmation(self) -> None:
        dialog = AnnotationDialog(np.zeros((100, 100, 3), dtype=np.uint8), ["object"])
        dialog.annotations.append(Annotation([10, 10, 50, 50]))
        with patch(
            "annotation.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            dialog.task_combo.setCurrentIndex(dialog.task_combo.findData("pose"))

        self.assertEqual(dialog.task, "pose")
        self.assertEqual(dialog.annotations, [])
        dialog.close()

    def test_annotation_list_selects_and_locks_target(self) -> None:
        dialog = AnnotationDialog(
            np.zeros((100, 100, 3), dtype=np.uint8), ["object"]
        )
        dialog.annotations.extend(
            [Annotation([5, 5, 45, 45]), Annotation([50, 50, 95, 95])]
        )
        dialog._sync_annotation_list()

        dialog.annotation_list.setCurrentRow(0)

        self.assertEqual(dialog.canvas.selected_box, 0)
        self.assertTrue(dialog.canvas.selection_locked)
        self.assertEqual(dialog.canvas._hit_box((70, 70)), -1)
        dialog.annotation_list.setCurrentRow(1)
        self.assertEqual(dialog.canvas.selected_box, 1)
        dialog.close()

    def test_classes_can_be_added_and_deleted_during_annotation(self) -> None:
        dialog = AnnotationDialog(
            np.zeros((100, 100, 3), dtype=np.uint8), ["object"]
        )
        classes = []
        dialog.classes_changed.connect(classes.append)
        with patch("annotation.QInputDialog.getText", return_value=("vehicle", True)):
            dialog._add_class()
        self.assertEqual(dialog.classes, ["object", "vehicle"])
        self.assertEqual(classes[-1], ["object", "vehicle"])

        dialog.annotations.append(Annotation([10, 10, 50, 50], class_id=1))
        dialog.class_combo.setCurrentIndex(1)
        with patch(
            "annotation.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            dialog._delete_class()

        self.assertEqual(dialog.classes, ["object"])
        self.assertEqual(dialog.annotations[0].class_id, 0)
        self.assertEqual(classes[-1], ["object"])
        dialog.close()

    def test_camera_snapshot_emits_rgb_frame(self) -> None:
        class FakeCapture:
            def isOpened(self):
                return True

            def read(self):
                return True, np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)

            def get(self, _property):
                return 30

            def release(self):
                pass

        with patch("capture_dialog.cv2.VideoCapture", return_value=FakeCapture()):
            dialog = CameraDialog()
            frames = []
            dialog.frame_captured.connect(frames.append)
            dialog.show()
            self.app.processEvents()
            dialog._read_frame()
            dialog.capture_frame()

        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0][0, 0], [30, 20, 10])
        self.assertTrue(dialog.isVisible())
        dialog.close()

    def test_missing_standard_streams_are_replaced_with_writable_sinks(self) -> None:
        stdout_sink = None
        stderr_sink = None
        try:
            with patch("main.sys.stdout", None), patch("main.sys.stderr", None):
                _ensure_standard_streams()
                stdout_sink = sys.stdout
                stderr_sink = sys.stderr
                self.assertIsNotNone(stdout_sink)
                self.assertIsNotNone(stderr_sink)
                stdout_sink.write("progress\n")
                stderr_sink.write("diagnostic\n")
        finally:
            if stdout_sink is not None:
                stdout_sink.close()
            if stderr_sink is not None:
                stderr_sink.close()

    def test_capture_hotkey_is_ignored_while_settings_dialog_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "main.keyboard.Listener", _FakeListener
        ), patch("main.mss.mss") as screen_factory:
            controller = AppController(self.app, Path(temporary), DEFAULT_SETTINGS.copy())
            controller.settings_open = True

            controller.begin_capture()

            screen_factory.assert_not_called()
            self.assertFalse(controller.is_capturing)
            controller.settings_open = False
            controller.quit()

    def test_pose_selection_resets_perspective_to_selected_box(self) -> None:
        from annotation import AnnotationDialog

        dialog = AnnotationDialog(
            np.zeros((500, 500, 3), dtype=np.uint8), ["object"], initial_task="pose"
        )
        dialog.annotations.extend(
            [
                Annotation([20, 30, 180, 230]),
                Annotation([240, 100, 440, 400]),
            ]
        )
        dialog.canvas.selected_box = 1
        dialog._selection_changed(1)

        np.testing.assert_allclose(
            dialog.source_points,
            [[240, 100], [440, 100], [440, 400], [240, 400]],
        )
        dialog.close()


if __name__ == "__main__":
    unittest.main()
