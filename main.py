from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from pathlib import Path


def _set_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        # Per-monitor V2：避免高缩放屏幕或多显示器不同缩放时截图出现偏移。
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        )
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def _ensure_standard_streams() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


_set_windows_dpi_awareness()
_ensure_standard_streams()

from PySide6.QtCore import QPoint, QRect, Qt, QSettings, QThread, QTimer, Signal, QObject
from PySide6.QtGui import QImageReader, QPixmap, QAction, QColor, QGuiApplication, QIcon, QKeyEvent, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMenu,
    QMessageBox,
    QStyle,
    QSystemTrayIcon,
    QWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QInputDialog,
)

import mss
import cv2
import numpy as np
from pynput import keyboard

from annotation import Annotation, AnnotationDialog, EditHistory
from capture_dialog import CameraDialog
from geometry import validate_quad
from guide_dialog import GUIDE_VERSION, GuideDialog
from modeling import export_onnx, TrainingStopped, prune_saved_models, train_model
from settings_dialog import SettingsDialog
from storage import encode_yolo_labels, save_sample, save_source_label
from training_dialog import TrainingDialog
from shortcuts import DEFAULT_SHORTCUTS
from naming import numbered_output


DEFAULT_SETTINGS = {
    "hotkey": "<ctrl>+<alt>+a",
    "shortcuts": dict(DEFAULT_SHORTCUTS),
    "dataset_dir": "datasets",
    "continuous_capture": False,
    "classes": ["object"],
    "magnifier_zoom": 3.0,
    "crosshair_color": [230, 235, 240],
    "box_color": [50, 220, 110],
    "magic_wand_tolerance": 20,
    "model_path": "models/detect/YOLO11n.pt",
    "model_paths": {
        "detect": "models/detect/YOLO11n.pt",
        "pose": "models/pose/yolo11n-pose.pt",
        "segment": "models/segment/yolo11n-seg.pt",
    },
    "max_models": 5,
    "model_dir": "models",
    "perspective_source_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "perspective_target_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "last_task": "detect",
}


def _application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_path(relative_path: str) -> Path:
    bundle_dir = Path(getattr(sys, "_MEIPASS", _application_dir()))
    return bundle_dir / relative_path


def _settings_path(app_dir: Path) -> Path:
    return app_dir / "config.json"


def _validate_rgb_color(name: str, value: object) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in value
        )
    ):
        raise ValueError(f"配置项 {name} 必须是 0 到 255 之间的 RGB 三元组")
    return list(value)


def _validate_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("配置文件根节点必须是 JSON 对象")
    hotkey = settings.get("hotkey")
    dataset_dir = settings.get("dataset_dir")
    continuous = settings.get("continuous_capture")
    classes = settings.get("classes")
    magnifier_zoom = settings.get("magnifier_zoom", DEFAULT_SETTINGS["magnifier_zoom"])
    crosshair_color = _validate_rgb_color(
        "crosshair_color",
        settings.get("crosshair_color", DEFAULT_SETTINGS["crosshair_color"]),
    )
    box_color = _validate_rgb_color(
        "box_color",
        settings.get("box_color", DEFAULT_SETTINGS["box_color"]),
    )
    magic_wand_tolerance = settings.get(
        "magic_wand_tolerance", DEFAULT_SETTINGS["magic_wand_tolerance"]
    )
    model_path = settings.get("model_path", DEFAULT_SETTINGS["model_path"])
    model_paths = settings.get("model_paths")
    if model_paths is None:
        model_paths = dict(DEFAULT_SETTINGS["model_paths"])
        model_paths["detect"] = model_path
    elif not isinstance(model_paths, dict):
        raise ValueError("配置项 model_paths 必须是任务到权重路径的对象")
    else:
        model_paths = {
            task: model_paths.get(task, default_path)
            for task, default_path in DEFAULT_SETTINGS["model_paths"].items()
        }
    model_dir = settings.get("model_dir", DEFAULT_SETTINGS["model_dir"])
    max_models = settings.get("max_models", DEFAULT_SETTINGS["max_models"])
    perspective_source_points = settings.get(
        "perspective_source_points", DEFAULT_SETTINGS["perspective_source_points"]
    )
    perspective_target_points = settings.get(
        "perspective_target_points", DEFAULT_SETTINGS["perspective_target_points"]
    )
    last_task = settings.get("last_task", DEFAULT_SETTINGS["last_task"])
    if not isinstance(hotkey, str) or not hotkey.strip():
        raise ValueError("配置项 hotkey 必须是非空字符串")
    if not isinstance(dataset_dir, str) or not dataset_dir.strip():
        raise ValueError("配置项 dataset_dir 必须是非空字符串")
    if not isinstance(continuous, bool):
        raise ValueError("配置项 continuous_capture 必须是布尔值")
    if (
        isinstance(magnifier_zoom, bool)
        or not isinstance(magnifier_zoom, (int, float))
        or not 2.0 <= magnifier_zoom <= 8.0
    ):
        raise ValueError("配置项 magnifier_zoom 必须是 2 到 8 之间的数字")
    if not isinstance(classes, list) or not classes:
        raise ValueError("配置项 classes 必须是非空且不重复的类别名称列表")
    normalized_classes = [
        name.strip() for name in classes if isinstance(name, str) and name.strip()
    ]
    if len(normalized_classes) != len(classes) or len(set(normalized_classes)) != len(classes):
        raise ValueError("配置项 classes 必须是非空且不重复的类别名称列表")
    if (
        isinstance(magic_wand_tolerance, bool)
        or not isinstance(magic_wand_tolerance, int)
        or not 0 <= magic_wand_tolerance <= 255
    ):
        raise ValueError("配置项 magic_wand_tolerance 必须是 0 到 255 之间的整数")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("配置项 model_path 必须是非空字符串")
    if any(not isinstance(path, str) or not path.strip() for path in model_paths.values()):
        raise ValueError("配置项 model_paths 必须为每种标注模式配置非空权重路径")
    if not isinstance(model_dir, str) or not model_dir.strip():
        raise ValueError("配置项 model_dir 必须是非空字符串")
    if isinstance(max_models, bool) or not isinstance(max_models, int) or max_models < 1:
        raise ValueError("配置项 max_models 必须是正整数")
    if last_task not in ("detect", "pose", "segment"):
        raise ValueError("配置项 last_task 必须是 detect、pose 或 segment")

    def validate_default_points(name: str, points: object) -> list[list[float]]:
        try:
            array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"配置项 {name} 必须是 4 个二维坐标") from error
        if array.shape != (4, 2) or not np.isfinite(array).all():
            raise ValueError(f"配置项 {name} 必须是 4 个二维坐标")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"配置项 {name} 的坐标必须在 0 到 1 之间")
        try:
            validate_quad(array)
        except ValueError as error:
            raise ValueError(f"配置项 {name} 无效：{error}") from error
        return array.tolist()

    perspective_source_points = validate_default_points(
        "perspective_source_points", perspective_source_points
    )
    perspective_target_points = validate_default_points(
        "perspective_target_points", perspective_target_points
    )
    keyboard.HotKey.parse(hotkey)
    shortcuts = settings.get("shortcuts")
    if shortcuts is None:
        shortcuts = dict(DEFAULT_SETTINGS["shortcuts"])
    elif not isinstance(shortcuts, dict):
        raise ValueError("配置项 shortcuts 必须是快捷键名称到按键序列的对象")
    else:
        shortcuts = {
            name: shortcuts.get(name, default_sequence)
            for name, default_sequence in DEFAULT_SETTINGS["shortcuts"].items()
        }
    seen_shortcuts = set()
    for name, sequence in shortcuts.items():
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"配置项 shortcuts.{name} 必须是非空字符串")
        parsed = QKeySequence.fromString(
            sequence, QKeySequence.SequenceFormat.PortableText
        )
        if not parsed.toString().strip():
            raise ValueError(f"配置项 shortcuts.{name} 不是有效的快捷键：{sequence}")
        canonical = parsed.toString(QKeySequence.SequenceFormat.PortableText)
        if canonical in seen_shortcuts:
            raise ValueError(f"快捷键重复：{sequence}")
        seen_shortcuts.add(canonical)
    return {
        "hotkey": hotkey.strip().lower(),
        "shortcuts": {name: sequence.strip() for name, sequence in shortcuts.items()},
        "dataset_dir": dataset_dir.strip(),
        "continuous_capture": continuous,
        "classes": normalized_classes,
        "magnifier_zoom": float(magnifier_zoom),
        "crosshair_color": crosshair_color,
        "box_color": box_color,
        "magic_wand_tolerance": magic_wand_tolerance,
        "model_path": model_path.strip(),
        "model_paths": {task: path.strip() for task, path in model_paths.items()},
        "max_models": max_models,
        "model_dir": model_dir.strip(),
        "perspective_source_points": perspective_source_points,
        "perspective_target_points": perspective_target_points,
        "last_task": last_task,
    }


def _load_settings(app_dir: Path) -> dict:
    path = _settings_path(app_dir)
    if not path.exists():
        _write_settings(app_dir, DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"读取配置文件失败：{error}") from error
    return _validate_settings(settings)


def _write_settings(app_dir: Path, settings: dict) -> None:
    path = _settings_path(app_dir)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


class TrainingWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)
    stopped = Signal()
    progress = Signal(int, int)
    log = Signal(str)

    def __init__(
        self,
        data_yaml: Path,
        model_path: Path,
        model_dir: Path,
        max_models: int,
        task: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str | None = None,
        workers: int = 8,
        patience: int = 100,
        lr0: float | None = None,
        optimizer: str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.model_dir = model_dir
        self.max_models = max_models
        self.task = task
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.workers = workers
        self.patience = patience
        self.lr0 = lr0
        self.optimizer = optimizer
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            model_path = train_model(
                self.data_yaml,
                self.model_path,
                self.model_dir,
                self.max_models,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                workers=self.workers,
                patience=self.patience,
                lr0=self.lr0,
                optimizer=self.optimizer,
                progress_callback=lambda epoch, total: self.progress.emit(epoch, total),
                log_callback=self.log.emit,
                should_stop=self._stop_event.is_set,
                expected_task=self.task,
            )
        except TrainingStopped:
            self.stopped.emit()
            return
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.completed.emit(self.task, str(model_path))


class OnnxWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, source: Path, destination: Path, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.destination = destination

    def run(self) -> None:
        try:
            export_onnx(self.source, self.destination)
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(str(self.destination))


class SelectionOverlay(QWidget):
    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self, desktop_rect: QRect, parent=None) -> None:
        super().__init__(parent)
        self.desktop_rect = desktop_rect
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        self._finished = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(desktop_rect)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 133))
        if self._start is not None and self._end is not None:
            selection = QRect(self._start, self._end).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawRect(selection)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(selection.topLeft() + QPoint(6, -6), "拖动选区 · Esc 取消")
        else:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(16, 28, "拖动鼠标选择截图区域 · Esc 取消")
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        self._end = event.position().toPoint()
        local_rect = QRect(self._start, self._end).normalized()
        if local_rect.width() < 2 or local_rect.height() < 2:
            self.close()
            return
        top_left = self.mapToGlobal(local_rect.topLeft())
        bottom_right = self.mapToGlobal(local_rect.bottomRight())
        global_rect = QRect(top_left, bottom_right).normalized()
        self._finished = True
        self.selected.emit(global_rect)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if not self._finished:
            self.cancelled.emit()
        super().closeEvent(event)


class AppController(QObject):
    capture_requested = Signal()

    def __init__(self, app: QApplication, app_dir: Path, settings: dict) -> None:
        super().__init__()
        self.app = app
        self.app_dir = app_dir
        self.settings = settings
        self.guide_storage = QSettings("LabelingCapture", "LabelingCapture")
        self.guide: GuideDialog | None = None
        self.listener: keyboard.Listener | None = None
        self.overlay: SelectionOverlay | None = None
        self.editor: AnnotationDialog | None = None
        self._background_editor: AnnotationDialog | None = None
        self.camera_dialog: CameraDialog | None = None
        self.onnx_worker: OnnxWorker | None = None
        self.image_files: list[str] = []
        self.image_index = -1
        self.label_files: dict[str, Path] = {}
        self.source_labels: dict[str, tuple[str, bytes | None]] = {}
        self.image_annotations: dict[str, EditHistory] = {}
        self.training_worker: TrainingWorker | None = None
        self.training_dialog: TrainingDialog | None = None
        self.desktop_rgb: np.ndarray | None = None
        self.desktop_origin = (0, 0)
        self.desktop_logical_origin = (0, 0)
        self.is_capturing = False
        self.settings_open = False
        self.capture_requested.connect(self.begin_capture)

        icon_path = _resource_path("picture/图标_128x128.ico")
        icon = QIcon(str(icon_path)) if icon_path.is_file() else QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_ComputerIcon
        )
        self.tray = QSystemTrayIcon(icon)
        self.tray.setToolTip("LabelingCapture")
        menu = QMenu()
        capture_action = QAction("截图标注", menu)
        capture_action.triggered.connect(self.begin_capture)
        camera_action = QAction("摄像机监视 / 截帧", menu)
        camera_action.triggered.connect(self.open_camera)
        settings_action = QAction("设置", menu)
        settings_action.triggered.connect(self.show_settings)
        folder_action = QAction("打开数据集目录", menu)
        folder_action.triggered.connect(self.open_dataset)
        self.train_action = QAction("手动训练模型…", menu)
        self.train_action.triggered.connect(self.start_training)
        exit_action = QAction("退出", menu)
        exit_action.triggered.connect(self.quit)
        menu.addAction(capture_action)
        menu.addAction(camera_action)
        menu.addAction(settings_action)
        menu.addAction(folder_action)
        menu.addAction(self.train_action)
        guide_action = QAction("使用引导", menu)
        guide_action.triggered.connect(self.show_guide)
        menu.addAction(guide_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()
        self._install_hotkey(settings["hotkey"])
        QTimer.singleShot(0, self.show_main)

    def show_main(self, *_args) -> None:
        if self.editor is None:
            self._open_annotation(np.zeros((360, 640, 3), dtype=np.uint8), workspace=True)
        if self.editor.isMinimized():
            self.editor.setWindowState(self.editor.windowState() & ~Qt.WindowState.WindowMinimized)
        self.editor.show()
        self.editor.raise_()
        self.editor.activateWindow()

    def show_first_run_guide(self) -> None:
        if self.guide_storage.value("guide/version", 0, type=int) < GUIDE_VERSION:
            self.show_guide()

    def show_guide(self, *_args) -> None:
        self.show_main()
        if self.guide is None:
            self.guide = GuideDialog(_resource_path("picture/guider.jpg"), self.editor)
            self.guide.finished.connect(self._finish_guide)
            self.guide.open()
        self.guide.raise_()
        self.guide.activateWindow()

    def _finish_guide(self, _result: int) -> None:
        self.guide_storage.setValue("guide/version", GUIDE_VERSION)
        self.guide_storage.sync()
        guide, self.guide = self.guide, None
        guide.deleteLater()

    def import_images(self, *_args) -> None:
        files, _ = QFileDialog.getOpenFileNames(None, "导入图片", "", "文件 (*.png *.jpg *.jpeg *.bmp)")
        if files:
            self._set_image_files(files)

    def import_dataset(self, *_args) -> None:
        folder = QFileDialog.getExistingDirectory(self.editor, "选择数据集目录")
        if not folder:
            return
        root = Path(folder)
        images = root / "images" if (root / "images").is_dir() else root
        image_folder = next((path for path in (images, *images.parents) if path.name.lower() == "images"), None)
        labels = (image_folder.with_name("labels") / images.relative_to(image_folder)
                  if image_folder is not None else root / "labels")
        files = sorted(
            str(path) for path in images.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not files:
            QMessageBox.warning(self.editor, "导入数据集", "目录中没有图片。")
            return
        paths = {file: (labels / Path(file).relative_to(images)).with_suffix(".txt") for file in files}
        if len({str(path).casefold() for path in paths.values()}) != len(paths):
            QMessageBox.warning(self.editor, "导入数据集", "同一目录下有同名图片，无法确定各自的标签文件。")
            return
        self._set_image_files(files, paths)

    def import_labels(self, *_args) -> None:
        if not self.editor.has_image:
            return
        path, _ = QFileDialog.getOpenFileName(self.editor, "导入 YOLO 标注框", "", "YOLO (*.txt)")
        if path and self.image_index >= 0:
            current = self.image_files[self.image_index]
            self._bind_source_label(current, Path(path))
        elif path:
            self._load_boxes(Path(path))

    def import_label_folder(self, *_args) -> None:
        if self.image_index < 0:
            QMessageBox.information(self.editor, "批量导入标注框", "请先导入图片或数据集。")
            return
        folder = QFileDialog.getExistingDirectory(self.editor, "选择 YOLO 标注目录")
        if not folder:
            return
        root = Path(folder)
        if (root / "labels").is_dir():
            root /= "labels"
        by_stem: dict[str, list[Path]] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".txt":
                by_stem.setdefault(path.stem.casefold(), []).append(path)
        matches = {}
        for file in self.image_files:
            image = Path(file)
            candidates = by_stem.get(image.stem.casefold(), [])
            if len(candidates) == 1:
                matches[file] = candidates[0]
            else:
                exact = [
                    path for path in candidates
                    if path.parent.relative_to(root).parts
                    and image.parent.parts[-len(path.parent.relative_to(root).parts):]
                    == path.parent.relative_to(root).parts
                ]
                if len(exact) == 1:
                    matches[file] = exact[0]
        if not matches:
            QMessageBox.warning(self.editor, "批量导入标注框", "没有找到与图片同名的标注文件。")
            return
        effective = {**self.label_files, **matches}
        if len({path.resolve() for path in effective.values()}) != len(effective):
            QMessageBox.warning(self.editor, "批量导入标注框", "有多张图片匹配到同一个标签文件，请检查目录结构。")
            return
        if not self._confirm_pending_edits(matches.keys()):
            return
        current = self.image_files[self.image_index]
        for file, path in matches.items():
            if file != current:
                self.image_annotations.pop(file, None)
                self.source_labels.pop(file, None)
                self.label_files[file] = path.resolve()
        if current in matches:
            self._bind_source_label(current, matches[current], confirm=False)
        QMessageBox.information(
            self.editor, "批量导入标注框",
            f"已匹配 {len(matches)}/{len(self.image_files)} 张图片。首次翻到该图片时会加载标注。",
        )

    def _bind_source_label(self, file: str, path: Path, *, confirm: bool = True) -> None:
        path = path.resolve()
        if any(other != file and path == label.resolve() for other, label in self.label_files.items()):
            QMessageBox.warning(self.editor, "导入标注框失败", "这个标签文件已关联到另一张图片。")
            return
        if confirm and not self._confirm_pending_edits((file,)):
            return
        if self._load_boxes(path, source_file=file):
            self.label_files[file] = path
            self.image_annotations.pop(file, None)
            self.editor.source_label_path = path
            self.editor._update_save_status()

    def _load_boxes(self, path: Path, *, source_file: str | None = None, missing_ok: bool = False) -> bool:
        editor = self.editor
        height, width = editor.image.shape[:2]
        boxes = []
        label_task = editor.task
        detected_format = None
        try:
            try:
                original = path.read_bytes()
            except FileNotFoundError:
                if not missing_ok:
                    raise
                original = None
            for line in (original or b"").decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                category, *fields = line.split()
                category = int(category)
                if not 0 <= category < len(editor.classes):
                    raise ValueError(f"无效的类别编号：{line}")
                values = np.array(fields, dtype=np.float64)
                if not np.isfinite(values).all() or not ((0 <= values) & (values <= 2)).all():
                    raise ValueError(f"无效的坐标：{line}")
                row_format = label_task
                if editor.task == "detect":
                    row_format = "obb" if len(values) == 8 else "detect"
                    if detected_format is not None and detected_format != row_format:
                        raise ValueError("标签文件混用了水平框和旋转框格式。")
                    detected_format = row_format
                if row_format in ("segment", "obb"):
                    minimum = 6 if row_format == "segment" else 8
                    if len(values) < minimum or len(values) % 2 or (values > 1).any():
                        raise ValueError(f"无效的角点坐标：{line}")
                    points = (values.reshape(-1, 2) * [width, height]).astype(np.float32)
                    if row_format == "obb":
                        (cx, cy), (w, h), angle = cv2.minAreaRect(points)
                        if not w or not h:
                            raise ValueError(f"无效的旋转框：{line}")
                        boxes.append(Annotation([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                                                category, angle=angle))
                    else:
                        left, top = points.min(axis=0)
                        right, bottom = points.max(axis=0)
                        boxes.append(Annotation(
                            [float(left), float(top), float(right), float(bottom)], category,
                            polygon=[tuple(point) for point in points.tolist()],
                        ))
                    continue
                x, y, w, h = values[:4]
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"无效的矩形框：{line}")
                box = Annotation([
                    max(0, x - w / 2) * width, max(0, y - h / 2) * height,
                    min(1, x + w / 2) * width, min(1, y + h / 2) * height,
                ], category)
                if row_format == "pose":
                    if len(values) <= 4 or (len(values) - 4) % 3:
                        raise ValueError("姿态标签应包含 x、y、可见度三列。")
                    for px, py, visibility in values[4:].reshape(-1, 3):
                        if px > 1 or py > 1 or visibility not in (0, 1, 2):
                            raise ValueError(f"无效的关键点：{line}")
                        box.keypoints.append((px * width, py * height) if visibility else None)
                        box.keypoint_visibility.append(int(visibility))
                elif len(values) != 4:
                    raise ValueError(f"无效的检测标签：{line}")
                boxes.append(box)
        except (OSError, ValueError, cv2.error) as error:
            QMessageBox.warning(editor, "导入标注框失败", str(error))
            return False
        if source_file is not None:
            self.source_labels[source_file] = (detected_format or label_task, original)
            editor.annotations.clear()
            editor.canvas.selected_box = -1
            editor.canvas.selected_keypoint = None
            editor.canvas.selection_locked = False
            editor.canvas.selection_changed.emit(-1)
        editor.annotations.extend(boxes)
        if source_file is not None:
            editor.reset_history(has_saved_label=original is not None)
        editor.canvas.changed.emit()
        editor.canvas.update()
        return True

    def import_model(self, *_args) -> None:
        path, _ = QFileDialog.getOpenFileName(self.editor, "标注工具", "", "文件 (*.pt *.onnx)")
        if path and self.editor:
            task = self.editor.task
            self.settings.setdefault("model_paths", {})[task] = path
            self.editor.model_path[task] = Path(path)

    def export_onnx(self, *_args) -> None:
        if self.onnx_worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self.editor, "选择 PT 模型", "", "PyTorch (*.pt)")
        if not path:
            return
        source = Path(path)
        output, _ = QFileDialog.getSaveFileName(
            self.editor, "保存 ONNX 模型", str(source.with_suffix(".onnx")), "ONNX (*.onnx)"
        )
        if not output:
            return
        destination = Path(output)
        if destination.suffix.lower() != ".onnx":
            destination = destination.with_suffix(".onnx")
            if destination.exists() and QMessageBox.question(
                self.editor, "覆盖模型", f"覆盖 {destination}？"
            ) != QMessageBox.StandardButton.Yes:
                return
        worker = OnnxWorker(source, destination, self)
        self.onnx_worker = worker
        worker.completed.connect(self._onnx_completed)
        worker.failed.connect(self._onnx_failed)
        worker.finished.connect(self._onnx_finished)
        self.editor.export_button.setEnabled(False)
        self.editor.export_button.setText("正在导出…")
        worker.start()

    def _onnx_completed(self, destination: str) -> None:
        QMessageBox.information(self.editor, "ONNX 导出完成", destination)

    def _onnx_failed(self, error: str) -> None:
        QMessageBox.critical(self.editor, "ONNX 导出失败", error)

    def _onnx_finished(self) -> None:
        worker = self.onnx_worker
        self.onnx_worker = None
        if self.editor is not None and hasattr(self.editor, "export_button"):
            self.editor.export_button.setText("导出 ONNX")
            self.editor.export_button.setEnabled(True)
        worker.deleteLater()

    def import_video(self, *_args) -> None:
        path, _ = QFileDialog.getOpenFileName(self.editor, "标注工具", "", "文件 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        interval, ok = QInputDialog.getInt(self.editor, "标注工具", "每隔多少帧提取一张图片？", 15, 1, 100000)
        if not ok:
            return
        from datetime import datetime
        task = self.editor.task
        video_root = self.dataset_dir / {"segment": "seg"}.get(task, task)
        with numbered_output(video_root, "video_") as video_id:
            destination = video_root / video_id / "images"
            destination.mkdir(parents=True, exist_ok=False)
        capture = cv2.VideoCapture(path)
        files = []
        try:
            if not capture.isOpened():
                raise ValueError("无法打开视频")
            destination.mkdir(parents=True, exist_ok=True)
            index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if index % interval == 0:
                    target = destination / f"frame_{index:08d}.png"
                    encoded, data = cv2.imencode(".png", frame)
                    if not encoded:
                        raise ValueError("无法编码视频帧")
                    data.tofile(target)
                    files.append(str(target))
                index += 1
        except Exception as error:
            QMessageBox.critical(self.editor, "标注工具", str(error))
        finally:
            capture.release()
        if files:
            self._set_image_files(files)

    def _set_image_files(self, files: list[str], labels: dict[str, Path] | None = None) -> None:
        first = self._find_readable_image(files, range(len(files)))
        if first is None or not self._confirm_pending_edits():
            return
        self.image_files = files
        self.label_files = dict(labels or {})
        self.source_labels.clear()
        self.image_annotations.clear()
        index, image = first
        self._display_image(index, image)

    def _find_readable_image(self, files: list[str], indexes: range) -> tuple[int, np.ndarray] | None:
        skipped = []
        result = None
        for index in indexes:
            image = self._read_image(files[index], warn=False)
            if image is not None:
                result = (index, image)
                break
            skipped.append(Path(files[index]).name)
        if skipped:
            names = "、".join(skipped[:5])
            QMessageBox.warning(self.editor, "无法读取图片", f"已跳过 {len(skipped)} 张无法读取的图片：{names}")
        return result

    def _read_image(self, file: str, *, warn: bool = True) -> np.ndarray | None:
        try:
            image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except (OSError, ValueError, cv2.error):
            pass
        if warn:
            QMessageBox.warning(self.editor, "无法读取图片", file)
        return None

    def navigate_image(self, step: int) -> None:
        if step not in (-1, 1):
            return
        stop = len(self.image_files) if step > 0 else -1
        target = self._find_readable_image(self.image_files, range(self.image_index + step, stop, step))
        if target is None:
            return
        index, image = target
        if self.image_index >= 0:
            current = self.image_files[self.image_index]
            self.editor.finish_edit()
            self.image_annotations[current] = self.editor.edit_history
        self._display_image(index, image)

    def _display_image(self, index: int, image: np.ndarray) -> None:
        editor = self.editor
        self.image_index = index
        file = self.image_files[index]
        editor.set_workspace_image(image)
        editor.source_label_path = self.label_files.get(file)
        cached = self.image_annotations.get(file)
        if cached is not None:
            editor.restore_history(cached)
        elif editor.source_label_path is not None:
            self._load_boxes(editor.source_label_path, source_file=file, missing_ok=True)
        editor._update_save_status()
        editor.setWindowTitle(f"LabelingCapture · {Path(file).name} · {index + 1}/{len(self.image_files)}")
        self._refresh_neighbors()

    def _confirm_pending_edits(self, files=None) -> bool:
        editor = self.editor
        if editor is None or not getattr(editor, "persistent_workspace", False) or self.image_index < 0:
            return True

        editor.finish_edit()
        original_index = self.image_index
        current = self.image_files[original_index]
        self.image_annotations[current] = editor.edit_history
        scope = self.image_files if files is None else files
        indexes = {file: index for index, file in enumerate(self.image_files)}
        pending = [
            indexes[file] for file in scope
            if file in self.image_annotations and self.image_annotations[file].dirty
        ]
        if not pending:
            return True
        button = QMessageBox.question(
            editor,
            "未保存的标注",
            f"还有 {len(pending)} 张图片有未保存的修改。是否保存？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if button == QMessageBox.StandardButton.Discard:
            return True
        if button != QMessageBox.StandardButton.Save:
            return False

        original_image = editor.image
        success = True
        for index in pending:
            if index != self.image_index:
                image = original_image if index == original_index else self._read_image(self.image_files[index])
                if image is None:
                    success = False
                    break
                self._display_image(index, image)
            editor._request_save()
            if editor.edit_history.dirty:
                success = False
                break
        if self.image_index != original_index:
            self._display_image(original_index, original_image)
        return success

    def _refresh_neighbors(self) -> None:
        editor = self.editor
        for step, button, preview in editor.neighbors:
            index = self.image_index + step
            available = self.image_index >= 0 and 0 <= index < len(self.image_files)
            button.setEnabled(available)
            preview.clear()
            preview.setToolTip("")
            if not available:
                preview.setText("—")
                continue
            file = self.image_files[index]
            reader = QImageReader(file)
            reader.setAutoTransform(True)
            if reader.size().isValid():
                reader.setScaledSize(reader.size().scaled(
                    preview.size(), Qt.AspectRatioMode.KeepAspectRatio
                ))
            thumbnail = reader.read()
            if thumbnail.isNull():
                preview.setText(Path(file).name)
            else:
                preview.setPixmap(QPixmap.fromImage(thumbnail))
            preview.setToolTip(Path(file).name)

    @property
    def dataset_dir(self) -> Path:
        path = Path(self.settings["dataset_dir"])
        return path if path.is_absolute() else self.app_dir / path

    def _install_hotkey(self, hotkey_text: str) -> None:
        parsed_hotkey = keyboard.HotKey.parse(hotkey_text)
        hotkey = keyboard.HotKey(parsed_hotkey, self.capture_requested.emit)
        listener: keyboard.Listener
        listener = keyboard.Listener(
            on_press=lambda key: hotkey.press(listener.canonical(key)),
            on_release=lambda key: hotkey.release(listener.canonical(key)),
        )
        try:
            listener.start()
        except Exception as error:
            raise RuntimeError(f"无法注册全局热键 {hotkey_text}：{error}") from error
        previous_listener = self.listener
        self.listener = listener
        if previous_listener is not None:
            previous_listener.stop()
            previous_listener.join(timeout=1)
        self.tray.setToolTip(f"LabelingCapture · {hotkey_text}")

    def begin_capture(self, *_args) -> None:
        if self.settings_open or self.is_capturing:
            return
        if self.editor is not None and not getattr(self.editor, "persistent_workspace", False):
            return
        try:
            with mss.mss() as screen:
                monitor = screen.monitors[0]
                screenshot = np.asarray(screen.grab(monitor))
            self.desktop_rgb = screenshot[:, :, :3][:, :, ::-1].copy()
        except Exception as error:
            QMessageBox.critical(None, "截图失败", f"无法读取桌面画面：{error}")
            return

        self.desktop_origin = (monitor["left"], monitor["top"])
        virtual_geometry = QGuiApplication.primaryScreen().virtualGeometry()
        self.desktop_logical_origin = (
            virtual_geometry.left(),
            virtual_geometry.top(),
        )
        desktop_rect = QRect(virtual_geometry)
        self.is_capturing = True
        self.overlay = SelectionOverlay(desktop_rect)
        self.overlay.selected.connect(self._selection_made)
        self.overlay.cancelled.connect(self._selection_cancelled)
        self.overlay.show()
        self.overlay.activateWindow()
        self.overlay.raise_()

    @staticmethod
    def _screen_dpr(rect: QRect) -> float:
        screen = QGuiApplication.screenAt(rect.center())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio())

    def _selection_made(self, global_rect: QRect) -> None:
        self.is_capturing = False
        overlay = self.overlay
        self.overlay = None
        if overlay is not None:
            overlay.deleteLater()
        if self.desktop_rgb is None:
            return
        dpr = self._screen_dpr(global_rect)
        logical_origin = getattr(self, "desktop_logical_origin", (0, 0))
        x0 = round((global_rect.left() - logical_origin[0]) * dpr)
        y0 = round((global_rect.top() - logical_origin[1]) * dpr)
        x1 = round((global_rect.right() - logical_origin[0]) * dpr)
        y1 = round((global_rect.bottom() - logical_origin[1]) * dpr)
        height, width = self.desktop_rgb.shape[:2]
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        crop = self.desktop_rgb[y0:y1, x0:x1].copy()
        self.desktop_rgb = None
        self._open_capture_editor(crop)

    def _open_annotation(self, image_rgb: np.ndarray, *, workspace: bool = False, task: str | None = None) -> None:
        self.editor = AnnotationDialog(
            image_rgb,
            self.settings["classes"],
            magnifier_zoom=self.settings.get("magnifier_zoom", 3.0),
            crosshair_color=self.settings.get("crosshair_color", (230, 235, 240)),
            box_color=self.settings.get("box_color", (50, 220, 110)),
            magic_wand_tolerance=self.settings.get("magic_wand_tolerance", 20),
            shortcuts=self.settings.get("shortcuts"),
            model_path={
                task: self._resolved_model_path(task)
                for task in ("detect", "pose", "segment")
            },
            perspective_source_defaults=self.settings.get(
                "perspective_source_points",
                DEFAULT_SETTINGS["perspective_source_points"],
            ),
            perspective_target_defaults=self.settings.get(
                "perspective_target_points",
                DEFAULT_SETTINGS["perspective_target_points"],
            ),
            initial_task=task or self.settings.get("last_task", "detect"),
        )
        if workspace:
            self._configure_workspace()
        self.editor.task_changed.connect(self._remember_task)
        self.editor.classes_changed.connect(self._remember_classes)
        self.editor.save_requested.connect(self._save_sample)
        self.editor.finished.connect(
            lambda _result, editor=self.editor: self._editor_closed(editor)
        )
        self.editor.show()
        self.editor.raise_()
        self.editor.activateWindow()

    def _configure_workspace(self) -> None:
        editor = self.editor
        editor.persistent_workspace = True
        editor.confirm_close = self._confirm_pending_edits
        editor.can_delete_class = lambda: not self.image_files
        editor.setWindowFlags(
            editor.windowFlags()
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        editor.setSizeGripEnabled(True)
        editor.setWindowTitle("LabelingCapture")
        editor.set_workspace_image(None)
        toolbar = QHBoxLayout()
        actions = QHBoxLayout()
        for title, callback in (
            ("导入数据集", self.import_dataset),
            ("导入图片", self.import_images),
            ("导入视频", self.import_video),
            ("单个标注", self.import_labels),
            ("批量标注", self.import_label_folder),
            ("导入模型", self.import_model),
            ("导出 ONNX", self.export_onnx),
            ("设置", self.show_settings),
            ("训练", self.start_training),
            ("摄像机", self.open_camera),
            ("截图", self.begin_capture),
            ("使用引导", self.show_guide),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row = toolbar if callback in (
                self.import_dataset, self.import_images, self.import_video,
                self.import_labels, self.import_label_folder, self.import_model,
            ) else actions
            row.addWidget(button)
            if callback == self.export_onnx:
                editor.export_button = button
                button.setEnabled(self.onnx_worker is None)
        editor.layout().insertLayout(0, toolbar)
        editor.layout().insertLayout(1, actions)

        navigation = QHBoxLayout()
        navigation.setSpacing(12)
        navigation.addStretch()
        editor.neighbors = []
        for step in (-1, 1):
            button = QPushButton()
            button.clicked.connect(lambda _checked=False, step=step: self.navigate_image(step))
            preview = QLabel()
            preview.setFixedSize(128, 64)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setStyleSheet("background: #e7ebf2; color: #4b586f; border-radius: 6px;")
            navigation.addWidget(button)
            navigation.addWidget(preview)
            editor.neighbors.append((step, button, preview))
        navigation.addStretch()
        editor.layout().insertLayout(2, navigation)
        self._configure_navigation_keys(self.settings.get("shortcuts", {}))
        self._refresh_neighbors()

    def _configure_navigation_keys(self, shortcuts: dict) -> None:
        from PySide6.QtGui import QShortcut

        editor = self.editor
        for shortcut in getattr(editor, "navigation_keys", []):
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        editor.navigation_keys = []
        keys = {**DEFAULT_SETTINGS["shortcuts"], **shortcuts}
        for (_, button, _), name, title in zip(
            editor.neighbors, ("previous_image", "next_image"), ("上一张", "下一张")
        ):
            sequence = keys[name]
            button.setText(f"{title} ({sequence})")
            shortcut = QShortcut(QKeySequence(sequence), editor)
            shortcut.activated.connect(button.click)
            editor.navigation_keys.append(shortcut)

    def open_camera(self, *_args) -> None:
        if self.is_capturing:
            QMessageBox.information(None, "摄像机", "请先完成或关闭当前操作。")
            return
        if self.camera_dialog is not None:
            self.camera_dialog.raise_()
            self.camera_dialog.activateWindow()
            return
        task = self.editor.task if self.editor is not None else self.settings.get("last_task", "detect")
        base = self.app_dir / self.settings.get("model_dir", "models") / task
        latest = max(base.glob("train_*.pt"), key=lambda path: path.stat().st_mtime, default=None)
        dialog = CameraDialog(
            model_path=latest or self._resolved_model_path(task),
            shortcuts=self.settings.get("shortcuts"),
            classes=self.settings["classes"],
            task=task,
        )
        dialog.frame_captured.connect(self._open_camera_frame)
        dialog.predicted_frame_captured.connect(self._open_camera_frame)
        dialog.finished.connect(
            lambda _result, camera=dialog: self._camera_dialog_closed(camera)
        )
        self.camera_dialog = dialog
        dialog.show()

    def _open_camera_frame(self, image_rgb: np.ndarray, predictions=None, task=None) -> None:
        if self.editor is not None and not getattr(self.editor, "persistent_workspace", False):
            QMessageBox.information(None, "摄像机", "请先完成或关闭当前标注，再截取下一帧。")
            return
        self._open_capture_editor(image_rgb, task=task)
        if predictions:
            self.editor.apply_predictions(predictions)

    def _open_capture_editor(self, image_rgb: np.ndarray, task: str | None = None) -> None:
        if self.editor is not None:
            self._background_editor = self.editor
            self._background_editor.hide()
        self._open_annotation(image_rgb, task=task)

    def _camera_dialog_closed(self, dialog: CameraDialog) -> None:
        if self.camera_dialog is dialog:
            self.camera_dialog = None
        dialog.deleteLater()

    def _remember_task(self, task: str) -> None:
        updated = dict(self.settings)
        updated["last_task"] = task
        try:
            _write_settings(self.app_dir, updated)
        except OSError as error:
            QMessageBox.warning(None, "标注模式", f"无法保存上次使用的模式：{error}")
            return
        self.settings = updated

    def _remember_classes(self, classes: list[str]) -> None:
        updated = dict(self.settings)
        updated["classes"] = list(classes)
        try:
            _write_settings(self.app_dir, updated)
        except OSError as error:
            QMessageBox.warning(None, "类别", f"无法保存类别设置：{error}")
            return
        self.settings = updated

    def _selection_cancelled(self) -> None:
        self.is_capturing = False
        overlay = self.overlay
        self.overlay = None
        if overlay is not None:
            overlay.deleteLater()
        self.desktop_rgb = None

    def _save_sample(self, payload: dict) -> None:
        if self.editor is None:
            return
        if self.editor.source_label_path is not None:
            self._save_dataset_label(payload)
            return
        task = payload["task"]
        rotated = task == "detect" and any(annotation.get("angle", 0) for annotation in payload["annotations"])
        output_task = "obb" if rotated else task
        directory = {"segment": "seg"}.get(task, task)
        collection = "datasets_obb" if rotated else "datasets1"
        destination = self.dataset_dir / directory / collection
        try:
            sample_id = save_sample(
                destination,
                self.editor.image,
                output_task,
                payload["annotations"],
                payload["source_points"],
                payload["target_points"],
                payload["homography"],
                self.settings["classes"],
                payload["perspective_target"],
            )
        except (OSError, ValueError, cv2.error) as error:
            QMessageBox.critical(self.editor, "保存失败", f"数据未能完整保存：{error}")
            return
        self.tray.showMessage(
            "LabelingCapture", f"已保存 {sample_id}", QSystemTrayIcon.MessageIcon.Information, 2500
        )
        continue_capture = self.settings["continuous_capture"]
        persistent = getattr(self.editor, "persistent_workspace", False)
        self.editor.edit_history.sample_label_path = destination / "labels" / f"{sample_id}.txt"
        self.editor.mark_saved()
        if continue_capture and not persistent:
            QTimer.singleShot(200, self.begin_capture)

    def _save_dataset_label(self, payload: dict) -> None:
        editor = self.editor
        file = self.image_files[self.image_index]
        path = editor.source_label_path
        if file not in self.source_labels:
            QMessageBox.warning(editor, "保存失败", "源标签未能读取。请检查标签内容和当前任务模式后重新导入。")
            return
        original_task, expected = self.source_labels[file]
        task = payload["task"]
        rotated = task == "detect" and any(item.get("angle", 0) for item in payload["annotations"])
        export_task = "obb" if rotated or (task == "detect" and original_task == "obb") else task
        if expected and original_task != export_task:
            QMessageBox.warning(editor, "保存失败", f"源标签使用 {original_task} 格式，当前标注使用 {export_task} 格式。")
            return
        try:
            height, width = editor.image.shape[:2]
            content = encode_yolo_labels(payload["annotations"], export_task, width, height)
            original = save_source_label(path, expected, content)
        except (ValueError, OSError) as error:
            QMessageBox.warning(editor, "保存源标签失败", str(error))
            return
        self.source_labels[file] = (export_task, original)
        editor.mark_saved()
        self.tray.showMessage("LabelingCapture", f"已保存源标签 {path.name}", QSystemTrayIcon.MessageIcon.Information, 2500)

    def _editor_closed(self, editor: AnnotationDialog) -> None:
        if self.editor is editor:
            self.editor = None
            if getattr(editor, "persistent_workspace", False):
                self.image_files.clear()
                self.image_index = -1
                self.label_files.clear()
                self.source_labels.clear()
                self.image_annotations.clear()
        editor.deleteLater()
        if self.editor is None and self._background_editor is not None:
            self.editor = self._background_editor
            self._background_editor = None
            self.editor.show()
        if not editor._saved:
            self.desktop_rgb = None

    def show_settings(self, *_args) -> None:
        if self.settings_open:
            return
        self.settings_open = True
        try:
            dialog = SettingsDialog(self.settings, self.app_dir, parent=self.editor)
            while dialog.exec() == QDialog.DialogCode.Accepted:
                try:
                    updated = dialog.values()
                    if (
                        self.editor is not None
                        and (self.editor.annotations or self.image_annotations or self.label_files)
                        and updated["classes"] != self.editor.classes
                    ):
                        raise ValueError("请先完成当前数据集标注，再修改类别列表，以免改变标注的类别编号。")
                    keyboard.HotKey.parse(updated["hotkey"])
                    self._install_hotkey(updated["hotkey"])
                    try:
                        _write_settings(self.app_dir, updated)
                    except OSError:
                        self._install_hotkey(self.settings["hotkey"])
                        raise
                    if self.editor is not None:
                        self._apply_editor_settings(updated)
                    self.settings = updated
                    if self.editor is not None:
                        self.editor.model_path = {
                            task: self._resolved_model_path(task)
                            for task in ("detect", "pose", "segment")
                        }
                    model_dir = Path(updated["model_dir"])
                    if not model_dir.is_absolute():
                        model_dir = self.app_dir / model_dir
                    for task in ("detect", "pose", "segment"):
                        try:
                            prune_saved_models(
                                model_dir / task,
                                updated["max_models"],
                                self._resolved_model_path(task),
                            )
                        except (OSError, ValueError) as error:
                            QMessageBox.warning(
                                None,
                                "模型清理失败",
                                f"无法清理 {task} 模型：{error}",
                            )
                    self.tray.showMessage(
                        "LabelingCapture",
                        "设置已保存",
                        QSystemTrayIcon.MessageIcon.Information,
                        1800,
                    )
                    return
                except (OSError, ValueError, RuntimeError) as error:
                    QMessageBox.warning(dialog, "设置无效", str(error))
        finally:
            self.settings_open = False

    def _apply_editor_settings(self, settings: dict) -> None:
        if self.camera_dialog is not None:
            self.camera_dialog.shortcuts = settings["shortcuts"]
            self.camera_dialog.shortcut_hint.setText(f"按 {settings['shortcuts']['camera_capture']} 截帧；录制文件使用 MP4 编码。")
        editor = self.editor
        editor.configure_shortcuts(settings["shortcuts"])
        if hasattr(editor, "neighbors"):
            self._configure_navigation_keys(settings["shortcuts"])
        editor.magnifier_zoom = settings["magnifier_zoom"]
        editor.canvas.magic_wand_tolerance = settings["magic_wand_tolerance"]
        editor.box_color = QColor(*settings["box_color"])
        editor.canvas.box_color = editor.box_color
        editor.canvas.crosshair_color = QColor(*settings["crosshair_color"])
        editor.preview_magnifier.set_crosshair_color(settings["crosshair_color"])
        if editor.classes != settings["classes"]:
            editor.classes = list(settings["classes"])
            editor.class_combo.blockSignals(True)
            editor.class_combo.clear()
            editor.class_combo.addItems(editor.classes)
            editor.class_combo.blockSignals(False)
            editor.canvas.class_count = len(editor.classes)
            editor.canvas.set_class(0)
        editor.source_defaults = np.asarray(settings["perspective_source_points"], dtype=np.float32)
        editor.target_defaults = np.asarray(settings["perspective_target_points"], dtype=np.float32)
        editor._refresh_preview()
        editor.canvas.update()

    def _resolved_model_path(self, task: str | None = None) -> Path:
        if task is None:
            configured_path = self.settings.get(
                "model_path", DEFAULT_SETTINGS["model_path"]
            )
        else:
            configured_path = self.settings.get("model_paths", {}).get(
                task, DEFAULT_SETTINGS["model_paths"][task]
            )
        path = Path(configured_path)
        if path.is_absolute():
            return path
        application_path = self.app_dir / path
        if application_path.is_file():
            return application_path
        bundle_path = Path(getattr(sys, "_MEIPASS", self.app_dir)) / path
        return bundle_path if bundle_path.is_file() else application_path

    def start_training(self, *_args) -> None:
        dialog = self.training_dialog
        if dialog is None:
            dialog = TrainingDialog(
                self.dataset_dir,
                initial_task=self.settings.get("last_task", "detect"),
            )
            dialog.start_requested.connect(self._begin_training)
            dialog.stop_requested.connect(self._stop_training)
            self.training_dialog = dialog
        dialog.showNormal()
        dialog.raise_()
        dialog.activateWindow()

    def _begin_training(self, config: dict) -> None:
        if self.training_worker is not None and self.training_worker.isRunning():
            QMessageBox.information(None, "模型训练", "已有训练任务正在运行。")
            return
        dialog = self.training_dialog
        task = config["task"]
        data_yaml = Path(config["data_yaml"])
        if not data_yaml.is_file():
            QMessageBox.warning(dialog, "训练配置", f"找不到数据集配置：{data_yaml}")
            return
        from modeling import training_model_path
        try:
            model_path = training_model_path(
                config.get("model_family", "yolo26"), config.get("model_scale", "n"),
                task, config.get("custom_model", ""),
            )
        except ValueError as error:
            QMessageBox.warning(dialog, "训练配置", str(error))
            return
        model_dir = Path(self.settings["model_dir"])
        if not model_dir.is_absolute():
            model_dir = self.app_dir / model_dir
        worker = TrainingWorker(
            data_yaml,
            model_path,
            model_dir / task,
            self.settings["max_models"],
            task,
            epochs=config.get("epochs", 100),
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 16),
            device=config.get("device"),
            workers=config.get("workers", 8),
            patience=config.get("patience", 100),
            lr0=config.get("lr0"),
            optimizer=config.get("optimizer", "auto"),
            parent=self,
        )
        worker.progress.connect(self._training_progress)
        worker.log.connect(self._training_log)
        worker.completed.connect(self._training_completed)
        worker.failed.connect(self._training_failed)
        worker.stopped.connect(self._training_stopped)
        worker.finished.connect(lambda worker=worker: self._training_finished(worker))
        self.training_worker = worker
        worker.start()
        if dialog is not None:
            dialog.set_training_started(
                task,
                f"模型：{model_path.name} · 数据集：{data_yaml.name} · epochs={config.get('epochs', 100)} "
                f"imgsz={config.get('imgsz', 640)} batch={config.get('batch', 16)}",
            )
        self.tray.showMessage(
            "模型训练", "训练已在后台开始；完成前仍使用当前模型进行预标注。",
            QSystemTrayIcon.MessageIcon.Information, 3000
        )

    def _stop_training(self) -> None:
        worker = self.training_worker
        if worker is not None and worker.isRunning():
            worker.stop()

    def _training_progress(self, epoch: int, total: int) -> None:
        if self.training_dialog is not None:
            self.training_dialog.update_progress(epoch, total)

    def _training_log(self, message: str) -> None:
        if self.training_dialog is not None:
            self.training_dialog.append_log(message)

    def _training_completed(self, task: str, model_path: str) -> None:
        updated = dict(self.settings)
        if task != "obb":
            updated["model_path"] = model_path
        updated["model_paths"] = dict(updated.get("model_paths", {}))
        updated["model_paths"][task] = model_path
        try:
            _write_settings(self.app_dir, updated)
        except OSError as error:
            QMessageBox.critical(
                None,
                "模型训练完成",
                f"新模型已保存，但无法更新活动模型配置：{error}",
            )
            return
        self.settings = updated
        for editor in (self.editor, self._background_editor):
            if editor is not None:
                editor.model_path[task] = Path(model_path)
        model_dir = Path(updated["model_dir"])
        if not model_dir.is_absolute():
            model_dir = self.app_dir / model_dir
        try:
            prune_saved_models(
                model_dir / task, updated["max_models"], Path(model_path)
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                None,
                "模型清理失败",
                f"新模型已启用，但无法清理 {task} 类别超出上限的旧模型：{error}",
            )
        if self.training_dialog is not None:
            self.training_dialog.set_training_finished(
                f"训练完成，已切换到新模型：{Path(model_path).name}"
            )
        self.tray.showMessage(
            "模型训练完成", "已切换到新模型用于后续预标注。",
            QSystemTrayIcon.MessageIcon.Information, 3500
        )

    def _training_failed(self, message: str) -> None:
        if self.training_dialog is not None:
            self.training_dialog.set_training_finished(message, error=True)
        QMessageBox.critical(None, "模型训练失败", message)

    def _training_stopped(self) -> None:
        if self.training_dialog is not None:
            self.training_dialog.set_training_stopped("训练已终止，当前模型保持不变。")
        self.tray.showMessage(
            "模型训练", "训练已终止，当前模型保持不变。",
            QSystemTrayIcon.MessageIcon.Information, 3000
        )

    def _training_finished(self, worker: TrainingWorker) -> None:
        if self.training_worker is worker:
            self.training_worker = None

    def open_dataset(self, *_args) -> None:
        try:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            path = str(self.dataset_dir.resolve())
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except (OSError, subprocess.SubprocessError) as error:
            QMessageBox.critical(None, "打开目录失败", str(error))

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_main()

    def quit(self, *_args) -> None:
        if self.onnx_worker is not None and self.onnx_worker.isRunning():
            QMessageBox.information(self.editor, "ONNX 导出", "请等待模型导出完成后退出。")
            return
        if self.training_worker is not None and self.training_worker.isRunning():
            QMessageBox.information(None, "模型训练", "训练正在运行，请等待训练完成后再退出。")
            return
        if self.camera_dialog is not None and self.camera_dialog._inference is not None:
            if not self.camera_dialog._closing:
                self.camera_dialog.finished.connect(self.quit)
                self.camera_dialog.reject()
            return
        if self.camera_dialog is not None:
            self.camera_dialog.reject()
        if self.guide is not None:
            self.guide.reject()
        while self.editor is not None:
            if not self.editor.close():
                return
        if self.listener is not None:
            self.listener.stop()
            self.listener.join(timeout=1)
        if self.overlay is not None:
            self.overlay.close()
        if self.training_dialog is not None:
            self.training_dialog.close()
        self.tray.hide()
        self.app.quit()


def _application_stylesheet() -> str:
    return """
    QDialog, QMessageBox {
        background: #f5f6f8;
    }
    QGroupBox {
        border: 1px solid #d4d8de;
        border-radius: 6px;
        margin-top: 14px;
        padding-top: 8px;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #3a6ea5;
    }
    QPushButton {
        background: #e8edf3;
        border: 1px solid #c3ccd8;
        border-radius: 5px;
        padding: 5px 14px;
        min-height: 20px;
    }
    QPushButton:hover {
        background: #dce6f0;
    }
    QPushButton:pressed {
        background: #c8d6e5;
    }
    QPushButton:disabled {
        color: #9aa5b1;
        background: #eceff2;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
        background: #ffffff;
        border: 1px solid #c3ccd8;
        border-radius: 4px;
        padding: 3px 6px;
        selection-background-color: #3a6ea5;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus {
        border: 1px solid #3a6ea5;
    }
    QListWidget {
        background: #ffffff;
        border: 1px solid #c3ccd8;
        border-radius: 4px;
    }
    QListWidget::item:selected {
        background: #3a6ea5;
        color: #ffffff;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QProgressBar {
        border: 1px solid #c3ccd8;
        border-radius: 5px;
        background: #ffffff;
        text-align: center;
        min-height: 16px;
    }
    QProgressBar::chunk {
        background: #3a6ea5;
        border-radius: 4px;
    }
    QToolTip {
        background: #2c3036;
        color: #f0f2f5;
        border: 1px solid #2c3036;
        padding: 3px 6px;
    }
    """


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_application_stylesheet())
    app.setApplicationName("LabelingCapture")
    icon_path = _resource_path("picture/图标_128x128.ico")
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setQuitOnLastWindowClosed(False)
    app_dir = _application_dir()
    try:
        settings = _load_settings(app_dir)
    except (OSError, ValueError) as error:
        QMessageBox.critical(None, "配置错误", str(error))
        return 2
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "无法启动", "当前桌面环境未提供系统托盘。")
        return 2
    try:
        controller = AppController(app, app_dir, settings)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        QMessageBox.critical(
            None,
            "无法启动全局热键",
            f"{error}\n\nWindows/Linux 的全局热键需要系统输入权限；Linux Wayland 会话可能不支持。",
        )
        return 2
    QTimer.singleShot(0, controller.show_first_run_guide)
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False):
        os.environ["YOLO_AUTOINSTALL"] = "False"
    raise SystemExit(main())
