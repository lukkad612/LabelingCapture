from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QKeySequence
from shortcuts import DEFAULT_SHORTCUTS, SHORTCUTS
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, app_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LabelingCapture - 设置")
        self.setMinimumSize(540, 620)
        self._app_dir = app_dir
        self.hotkey_edit = QLineEdit(settings["hotkey"])
        shortcut_defaults = DEFAULT_SHORTCUTS
        configured_shortcuts = settings.get("shortcuts", shortcut_defaults)
        self.shortcut_edits: dict[str, QLineEdit] = {}
        for name, (label, _default) in SHORTCUTS.items():
            edit = QLineEdit(configured_shortcuts.get(name, shortcut_defaults[name]))
            self.shortcut_edits[name] = edit
        self.dataset_edit = QLineEdit(settings["dataset_dir"])
        browse_button = QPushButton("浏览…")
        browse_button.clicked.connect(self._browse_dataset)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        dataset_row.addWidget(browse_button)

        self.continuous_check = QCheckBox("保存后继续截图")
        self.continuous_check.setChecked(settings["continuous_capture"])

        self.magnifier_zoom = QDoubleSpinBox()
        self.magnifier_zoom.setRange(2.0, 8.0)
        self.magnifier_zoom.setSingleStep(0.5)
        self.magnifier_zoom.setSuffix("×")
        self.magnifier_zoom.setValue(float(settings.get("magnifier_zoom", 3.0)))
        self.crosshair_color = self._make_rgb_controls(
            settings.get("crosshair_color", [230, 235, 240])
        )
        self.box_color = self._make_rgb_controls(
            settings.get("box_color", [50, 220, 110])
        )
        self.magic_wand_tolerance = QSpinBox()
        self.magic_wand_tolerance.setRange(0, 255)
        self.magic_wand_tolerance.setValue(settings.get("magic_wand_tolerance", 20))
        self.model_path_edit = QLineEdit(settings.get("model_path", ""))
        model_browse = QPushButton("浏览…")
        model_browse.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(model_browse)
        configured_models = settings.get(
            "model_paths",
            {
                "detect": "models/detect/YOLO11n.pt",
                "pose": "models/pose/yolo11n-pose.pt",
                "segment": "models/segment/yolo11n-seg.pt",
            },
        )
        self.model_path_edits: dict[str, QLineEdit] = {}
        model_group = QGroupBox("按标注模式选择预标注模型")
        model_form = QFormLayout(model_group)
        for task in ("detect", "pose", "segment"):
            path_edit = QLineEdit(configured_models.get(task, ""))
            browse = QPushButton("浏览…")
            browse.clicked.connect(
                lambda _checked=False, selected_task=task: self._browse_task_model(
                    selected_task
                )
            )
            row = QHBoxLayout()
            row.addWidget(path_edit)
            row.addWidget(browse)
            model_form.addRow(task, row)
            self.model_path_edits[task] = path_edit
        self.max_models = QSpinBox()
        self.max_models.setRange(1, 1000)
        self.max_models.setValue(settings.get("max_models", 5))
        self.model_dir = settings.get("model_dir", "models")
        self.last_task = settings.get("last_task", "detect")
        self.perspective_source_points = self._make_point_controls(
            settings.get(
                "perspective_source_points",
                [[0, 0], [1, 0], [1, 1], [0, 1]],
            )
        )
        self.perspective_target_points = self._make_point_controls(
            settings.get(
                "perspective_target_points",
                [[0, 0], [1, 0], [1, 1], [0, 1]],
            )
        )

        self.class_list = QListWidget()
        self.class_list.addItems(settings["classes"])
        add_button = QPushButton("新增")
        edit_button = QPushButton("重命名")
        remove_button = QPushButton("删除")
        add_button.clicked.connect(self._add_class)
        edit_button.clicked.connect(self._edit_class)
        remove_button.clicked.connect(self._remove_class)
        class_buttons = QHBoxLayout()
        class_buttons.addWidget(add_button)
        class_buttons.addWidget(edit_button)
        class_buttons.addWidget(remove_button)

        form = QFormLayout()
        form.addRow("数据集目录", dataset_row)
        form.addRow("", self.continuous_check)
        form.addRow("标注放大镜倍率", self.magnifier_zoom)
        form.addRow("辅助十字 RGB", self._rgb_row(self.crosshair_color))
        form.addRow("标注框 RGB", self._rgb_row(self.box_color))
        form.addRow("魔棒颜色容差", self.magic_wand_tolerance)
        form.addRow("预标注模型", model_row)
        form.addRow("模式模型", model_group)
        form.addRow("最多保存模型数", self.max_models)

        save_button = QPushButton("保存")
        cancel_button = QPushButton("取消")
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(save_button)
        footer.addWidget(cancel_button)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._make_shortcuts_group())
        content_layout.addLayout(form)
        content_layout.addWidget(self._make_perspective_defaults_group())
        content_layout.addWidget(QLabel("类别（列表顺序即 YOLO 类别 ID）"))
        content_layout.addWidget(self.class_list)
        content_layout.addLayout(class_buttons)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addLayout(footer)

    def values(self) -> dict:
        dataset_dir = self.dataset_edit.text().strip()
        hotkey = self.hotkey_edit.text().strip().lower()
        classes = [
            self.class_list.item(index).text().strip()
            for index in range(self.class_list.count())
        ]
        if not dataset_dir:
            raise ValueError("数据集目录不能为空")
        if not hotkey:
            raise ValueError("全局热键不能为空")
        shortcuts = {}
        for name, control in self.shortcut_edits.items():
            sequence = control.text().strip()
            if not sequence:
                raise ValueError(f"快捷键“{name}”不能为空")
            parsed = QKeySequence.fromString(
                sequence, QKeySequence.SequenceFormat.PortableText
            )
            if not parsed.toString().strip():
                raise ValueError(f"快捷键“{name}”无效：{sequence}")
            if parsed in [QKeySequence(value) for value in shortcuts.values()]:
                raise ValueError(f"快捷键重复：{sequence}")
            shortcuts[name] = sequence
        if not classes or any(not name for name in classes):
            raise ValueError("至少需要一个非空类别")
        if len(set(classes)) != len(classes):
            raise ValueError("类别名称不能重复")
        return {
            "hotkey": hotkey,
            "shortcuts": shortcuts,
            "dataset_dir": dataset_dir,
            "continuous_capture": self.continuous_check.isChecked(),
            "classes": classes,
            "magnifier_zoom": self.magnifier_zoom.value(),
            "crosshair_color": [spin.value() for spin in self.crosshair_color],
            "box_color": [spin.value() for spin in self.box_color],
            "magic_wand_tolerance": self.magic_wand_tolerance.value(),
            "model_path": self.model_path_edit.text().strip(),
            "model_paths": {
                task: control.text().strip()
                for task, control in self.model_path_edits.items()
            },
            "max_models": self.max_models.value(),
            "model_dir": self.model_dir,
            "perspective_source_points": [
                [control.value() for control in row]
                for row in self.perspective_source_points
            ],
            "perspective_target_points": [
                [control.value() for control in row]
                for row in self.perspective_target_points
            ],
            "last_task": self.last_task,
        }

    @staticmethod
    def _make_rgb_controls(color: list[int]) -> list[QSpinBox]:
        controls = []
        for value in color:
            spin = QSpinBox()
            spin.setRange(0, 255)
            spin.setValue(value)
            controls.append(spin)
        return controls

    @staticmethod
    def _rgb_row(controls: list[QSpinBox]) -> QHBoxLayout:
        row = QHBoxLayout()
        for channel, control in zip(("R", "G", "B"), controls):
            row.addWidget(QLabel(channel))
            row.addWidget(control)
        row.addStretch(1)
        return row

    def _make_shortcuts_group(self) -> QGroupBox:
        group = QGroupBox("快捷键")
        form = QFormLayout(group)
        form.addRow("全局截图热键", self.hotkey_edit)
        for name, (label, _default) in SHORTCUTS.items():
            form.addRow(label, self.shortcut_edits[name])
        hint = QLabel(
            "快捷键使用 Qt 格式（如 Ctrl+S、Esc）；全局热键使用 pynput 格式（如 <ctrl>+<alt>+a）。"
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return group

    def _browse_dataset(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择数据集目录", str(self._resolved_dataset())
        )
        if selected:
            try:
                self.dataset_edit.setText(
                    str(Path(selected).resolve().relative_to(self._app_dir.resolve()))
                )
            except ValueError:
                self.dataset_edit.setText(selected)

    def _resolved_dataset(self) -> Path:
        path = Path(self.dataset_edit.text().strip())
        return path if path.is_absolute() else self._app_dir / path

    def _browse_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 权重",
            self.model_path_edit.text(),
            "YOLO 权重 (*.pt)",
        )
        if selected:
            self.model_path_edit.setText(selected)

    def _browse_task_model(self, task: str) -> None:
        control = self.model_path_edits[task]
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {task} 模型权重",
            control.text(),
            "YOLO 权重 (*.pt)",
        )
        if selected:
            control.setText(selected)

    @staticmethod
    def _make_point_controls(points: list[list[float]]) -> list[list[QDoubleSpinBox]]:
        rows: list[list[QDoubleSpinBox]] = []
        for point in points:
            row = []
            for value in point:
                control = QDoubleSpinBox()
                control.setDecimals(3)
                control.setRange(0.0, 1.0)
                control.setSingleStep(0.01)
                control.setValue(float(value))
                row.append(control)
            rows.append(row)
        return rows

    def _make_perspective_defaults_group(self) -> QGroupBox:
        group = QGroupBox("透视默认四点（归一化坐标 0–1）")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("点序"), 0, 0)
        for column, title in enumerate(("源 X", "源 Y", "目标 X", "目标 Y"), start=1):
            grid.addWidget(QLabel(title), 0, column)
        for index in range(4):
            grid.addWidget(QLabel(str(index + 1)), index + 1, 0)
            controls = (
                self.perspective_source_points[index]
                + self.perspective_target_points[index]
            )
            for column, control in enumerate(controls, start=1):
                grid.addWidget(control, index + 1, column)
        grid.addWidget(
            QLabel("点序：左上、右上、右下、左下。变换固定使用 4 对点。"),
            5,
            0,
            1,
            5,
        )
        return group

    def _add_class(self) -> None:
        name, accepted = QInputDialog.getText(self, "新增类别", "类别名称：")
        if accepted and name.strip():
            if self.class_list.findItems(name.strip(), Qt.MatchFlag.MatchExactly):
                QMessageBox.warning(self, "类别重复", "该类别名称已存在。")
                return
            self.class_list.addItem(name.strip())

    def _edit_class(self) -> None:
        item = self.class_list.currentItem()
        if item is None:
            return
        name, accepted = QInputDialog.getText(
            self, "重命名类别", "类别名称：", text=item.text()
        )
        if accepted and name.strip():
            matches = self.class_list.findItems(name.strip(), Qt.MatchFlag.MatchExactly)
            if matches and matches[0] is not item:
                QMessageBox.warning(self, "类别重复", "该类别名称已存在。")
                return
            item.setText(name.strip())

    def _remove_class(self) -> None:
        item = self.class_list.currentItem()
        if item is None:
            return
        if self.class_list.count() == 1:
            QMessageBox.warning(self, "无法删除", "至少保留一个类别。")
            return
        self.class_list.takeItem(self.class_list.row(item))
