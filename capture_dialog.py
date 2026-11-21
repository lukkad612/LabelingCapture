from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


from modeling import load_prediction_model, predict_annotations
from naming import numbered_output


class CameraInference(QThread):
    result = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, frame, path, classes, task, model, parent):
        super().__init__(parent)
        self.frame = frame
        self.path = path
        self.classes = classes
        self.task = task
        self.model = model

    def run(self) -> None:
        try:
            if self.model is None:
                self.model = load_prediction_model(self.path)
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            predictions = predict_annotations(rgb, self.path, self.classes, self.task, model=self.model)
            self.result.emit(rgb, predictions, self.model)
        except Exception as error:
            self.failed.emit(str(error))


class CameraDialog(QDialog):
    frame_captured = Signal(object)
    predicted_frame_captured = Signal(object, object, str)

    def __init__(self, parent=None, *, model_path=None, classes=None, task="detect", shortcuts=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("摄像机监视")
        self.shortcuts = shortcuts or {}
        self.resize(900, 650)
        self._capture: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._latest_frame: np.ndarray | None = None
        self._record_path: Path | None = None
        self._record_size: tuple[int, int] | None = None
        self._camera_index = 0
        self._backends: list[int] = []
        self._read_failures = 0
        self._sample_aspect = 1.0
        self._preview_image = QImage()
        self._model_path = model_path
        self._classes = list(classes or ["object"])
        self._task = task
        self._model = None
        self._inference = None
        self._inference_frame = None
        self._closing = False
        self._inference_enabled = False

        self.camera_combo = QComboBox()
        for index in range(8):
            self.camera_combo.addItem(f"摄像机 {index + 1}", index)
        self.camera_combo.currentIndexChanged.connect(self._open_camera)

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("自动", None)
        if os.name == "nt":
            self.backend_combo.addItem("DirectShow", cv2.CAP_DSHOW)
            self.backend_combo.addItem("Media Foundation", cv2.CAP_MSMF)
            self.backend_combo.currentIndexChanged.connect(
                lambda _index: self._open_camera(self.camera_combo.currentData())
            )
        self.aspect_combo = QComboBox()
        for label, aspect in (
            ("自动", None), ("16:9", 16 / 9), ("4:3", 4 / 3),
            ("9:16", 9 / 16), ("1:1", 1.0), ("原始像素", 0.0),
        ):
            self.aspect_combo.addItem(label, aspect)
        self.aspect_combo.setToolTip("自动使用摄像机报告的比例。虚拟摄像机比例异常时，可选择 OBS 的输出比例。")
        self.aspect_combo.currentIndexChanged.connect(self._reset_frame)
        self.frame_status = QLabel()

        self.preview = QLabel("正在连接摄像机…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setScaledContents(False)
        self.preview.setStyleSheet("background: #202226; color: #ddd;")

        snapshot_button = QPushButton("截取当前帧并标注")
        snapshot_button.clicked.connect(self.capture_frame)
        self.record_button = QPushButton("开始录制 MP4")
        self.record_button.clicked.connect(self._toggle_recording)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("摄像机"))
        toolbar.addWidget(self.camera_combo)
        self.model_button = QPushButton("选择模型")
        self.model_button.clicked.connect(self._choose_model)
        self.inference_button = QPushButton("实时推理")
        self.inference_button.setCheckable(True)
        self.inference_button.toggled.connect(self._toggle_inference)
        toolbar.addWidget(self.model_button)
        toolbar.addWidget(self.inference_button)
        toolbar.addStretch(1)
        toolbar.addWidget(snapshot_button)
        toolbar.addWidget(self.record_button)
        toolbar.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        format_row = QHBoxLayout()
        if os.name == "nt":
            format_row.addWidget(QLabel("采集接口"))
            format_row.addWidget(self.backend_combo)
        format_row.addWidget(QLabel("画面比例"))
        format_row.addWidget(self.aspect_combo)
        format_row.addWidget(self.frame_status)
        format_row.addStretch(1)
        layout.addLayout(format_row)
        layout.addWidget(self.preview, 1)
        self.inference_status = QLabel()
        self._show_model_name()
        layout.addWidget(self.inference_status)
        self.shortcut_hint = QLabel(f"按 {self.shortcuts.get('camera_capture', 'Space')} 截帧；录制文件使用 MP4 编码。")
        layout.addWidget(self.shortcut_hint)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._timer.start(30)
        self._open_camera(0)

    def _show_model_name(self) -> None:
        name = self._model_path.name if self._model_path else "未选择模型"
        self.inference_status.setText(f"推理模型：{name}（{self._task}）")

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择推理模型", "", "模型 (*.pt *.onnx)")
        if path:
            self.inference_button.setChecked(False)
            self._model_path = Path(path)
            self._model = None
            self._show_model_name()

    def _toggle_inference(self, enabled: bool) -> None:
        if enabled and (self._model_path is None or not self._model_path.is_file()):
            self.inference_button.setChecked(False)
            self.inference_status.setText("请选择有效的推理模型。")
            return
        self._inference_enabled = enabled
        if not enabled and self._inference is not None:
            self._inference.requestInterruption()
        self._inference_frame = None
        self._show_model_name()

    def _inference_ready(self, rgb, predictions, model) -> None:
        if self._closing or not self._inference_enabled or self._inference.isInterruptionRequested():
            return
        self._model = model
        self._inference_frame = rgb, predictions
        overlay = rgb.copy()
        for item in predictions:
            left, top, right, bottom = map(round, item["box"])
            cv2.rectangle(overlay, (left, top), (right, bottom), (80, 220, 130), 2)
            polygon = item.get("polygon", [])
            if polygon:
                cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (100, 200, 255), 2)
            for point in item.get("keypoints", []):
                if point is not None:
                    cv2.circle(overlay, tuple(map(round, point)), 3, (100, 200, 255), -1)
            cv2.putText(overlay, str(item["class_id"]), (left, max(14, top - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 130), 2)
        self._display_frame(overlay)

    def _inference_failed(self, error: str) -> None:
        if not self._inference.isInterruptionRequested():
            self.inference_button.setChecked(False)
            self.inference_status.setText(f"推理失败：{error}")

    def _inference_finished(self) -> None:
        worker = self._inference
        self._inference = None
        worker.deleteLater()
        if self._closing:
            self.reject()

    def _open_camera(self, index: int) -> None:
        if self._writer is not None:
            QMessageBox.information(self, "摄像机", "请先停止录制，再切换摄像机。")
            return
        self._release_camera()
        self._camera_index = index
        selected = self.backend_combo.currentData()
        if selected is not None:
            self._backends = [selected]
        else:
            self._backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_ANY]
        self._try_next_backend()

    def _try_next_backend(self) -> None:
        self._read_failures = 0
        while self._backends:
            backend = self._backends.pop(0)
            capture = cv2.VideoCapture(self._camera_index, backend)
            if capture.isOpened():
                self._capture = capture
                numerator = capture.get(cv2.CAP_PROP_SAR_NUM)
                denominator = capture.get(cv2.CAP_PROP_SAR_DEN)
                ratio = numerator / denominator if denominator > 0 else 0
                self._sample_aspect = ratio if 0.25 <= ratio <= 4 else 1.0
                return
            capture.release()
        self.preview.setText(f"无法读取摄像机 {self._camera_index + 1}，请检查设备或切换采集接口。")

    def _reset_frame(self, *_args) -> None:
        self._latest_frame = None
        self._inference_frame = None
        if self._inference is not None:
            self._inference.requestInterruption()
        self._preview_image = QImage()
        self.preview.clear()
        self.frame_status.clear()

    def _correct_aspect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        override = self.aspect_combo.currentData()
        target_width = max(
            1, round(width * self._sample_aspect if override is None else height * override)
        ) if override != 0 else width
        if target_width == width:
            return frame
        interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
        return cv2.resize(frame, (target_width, height), interpolation=interpolation)

    def _read_frame(self) -> None:
        if self._capture is None:
            return
        success, frame = self._capture.read()
        if not success or frame is None or not frame.size:
            if self._latest_frame is None:
                self._read_failures += 1
                if self._read_failures >= 15:
                    self._release_camera()
                    self._try_next_backend()
            return
        self._read_failures = 0
        input_height, input_width = frame.shape[:2]
        frame = self._correct_aspect(frame)
        height, width = frame.shape[:2]
        if self._latest_frame is not None and self._latest_frame.shape != frame.shape:
            self._reset_frame()
        self._latest_frame = frame.copy()
        if (input_width, input_height) == (width, height):
            self.frame_status.setText(f"{width} × {height}")
        else:
            self.frame_status.setText(f"{input_width} × {input_height} → {width} × {height}")
        if self._writer is not None:
            if self._record_size != (width, height):
                self._stop_recording()
            else:
                self._writer.write(frame)
        if self._inference_enabled:
            if self._inference is None:
                worker = CameraInference(frame, self._model_path, self._classes, self._task, self._model, self)
                self._inference = worker
                worker.result.connect(self._inference_ready)
                worker.failed.connect(self._inference_failed)
                worker.finished.connect(self._inference_finished)
                worker.start()
            if self._inference_frame is not None:
                return
        self._display_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _display_frame(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        self._preview_image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._fit_preview()

    def _fit_preview(self) -> None:
        if self._preview_image.isNull():
            return
        self.preview.setPixmap(
            QPixmap.fromImage(self._preview_image).scaled(
                self.preview.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_preview()

    def capture_frame(self) -> None:
        if self._latest_frame is None:
            QMessageBox.warning(self, "截帧失败", "当前没有可用的摄像机画面。")
            return
        if self._inference_enabled:
            if self._inference_frame is None:
                self.inference_status.setText("正在等待第一帧的推理结果。")
                return
            rgb, predictions = self._inference_frame
            self.predicted_frame_captured.emit(rgb.copy(), predictions, self._task)
            return
        frame_rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
        self.frame_captured.emit(frame_rgb)

    def _toggle_recording(self) -> None:
        if self._writer is not None:
            self._stop_recording()
            return
        if self._latest_frame is None:
            QMessageBox.warning(self, "录制失败", "当前没有可用的摄像机画面。")
            return
        with numbered_output(Path.cwd(), "camera_") as recording_id:
            selected, _ = QFileDialog.getSaveFileName(
                self, "保存摄像机视频", str(Path.cwd() / f"{recording_id}.mp4"), "MP4 视频 (*.mp4)"
            )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        height, width = self._latest_frame.shape[:2]
        fps = float(self._capture.get(cv2.CAP_PROP_FPS)) if self._capture else 0
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 30.0
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            QMessageBox.critical(self, "录制失败", f"无法创建 MP4 文件：{path}")
            return
        self._writer = writer
        self._record_path = path
        self._record_size = (width, height)
        self.camera_combo.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.aspect_combo.setEnabled(False)
        self.record_button.setText("停止录制")
        self.preview.setToolTip(f"正在录制：{path}")

    def _stop_recording(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.release()
        self._record_size = None
        self.camera_combo.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.aspect_combo.setEnabled(True)
        self.record_button.setText("开始录制 MP4")
        if self._record_path is not None:
            self.preview.setToolTip("")
            QMessageBox.information(
                self, "录制完成", f"视频已保存：{self._record_path}"
            )
            self._record_path = None

    def _release_camera(self) -> None:
        self._reset_frame()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.preview.setText("正在连接摄像机…")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if QKeySequence(event.keyCombination()) == QKeySequence(self.shortcuts.get("camera_capture", "Space")):
            self.capture_frame()
            return
        super().keyPressEvent(event)

    def done(self, result: int) -> None:
        self._timer.stop()
        self._closing = True
        if self._writer is not None:
            self._stop_recording()
        self._release_camera()
        if self._inference is not None:
            self.hide()
            return
        super().done(result)
