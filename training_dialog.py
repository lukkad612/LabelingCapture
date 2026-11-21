from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)


class TrainingDialog(QDialog):
    """训练配置与进度窗口。

    窗口在点击“开始训练”后保持打开；关闭窗口只是隐藏，训练在后台继续。
    """

    start_requested = Signal(dict)
    stop_requested = Signal()

    def __init__(
        self,
        dataset_dir: Path,
        initial_task: str = "detect",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowTitle("模型训练")
        self.setMinimumSize(560, 520)
        self.resize(760, 740)
        self._running = False

        self.data_edit = QLineEdit()
        data_browse = QPushButton("浏览…")
        data_browse.clicked.connect(self._browse_data)
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_edit)
        data_row.addWidget(data_browse)

        self.task_combo = QComboBox()
        for task in ("detect", "pose", "segment", "obb"):
            self.task_combo.addItem("obb（旋转框）" if task == "obb" else task, task)
        task_index = self.task_combo.findData(initial_task)
        if task_index >= 0:
            self.task_combo.setCurrentIndex(task_index)

        self.model_combo = QComboBox()
        for title, family in (("YOLO26（默认）", "yolo26"), ("YOLO11", "yolo11"),
                              ("YOLOv8", "yolov8"), ("自定义权重 / 架构", "custom")):
            self.model_combo.addItem(title, family)
        self.scale_combo = QComboBox()
        for scale in ("n", "s", "m", "l", "x"):
            self.scale_combo.addItem(scale, scale)
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("选择 .pt 权重或 .yaml 模型架构")
        self.custom_model_button = QPushButton("浏览…")
        self.custom_model_button.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 2)
        model_row.addWidget(QLabel("规模"))
        model_row.addWidget(self.scale_combo)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_model_edit)
        custom_row.addWidget(self.custom_model_button)
        self.model_hint = QLabel("预训练权重首次使用时需要联网下载；自定义 YAML 从头训练。")
        self.model_hint.setWordWrap(True)
        self.model_combo.currentIndexChanged.connect(self._update_model_controls)
        self._update_model_controls()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(8)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(100)
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(6)
        self.lr0_spin.setRange(0.000001, 1.0)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setValue(0.01)
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")
        self.optimizer_combo = QComboBox()
        for optimizer in (
            "auto",
            "SGD",
            "Adam",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
        ):
            self.optimizer_combo.addItem(optimizer, optimizer)

        config_group = QGroupBox("训练参数（开始训练时生效，可随时调整）")
        config_form = QFormLayout(config_group)
        config_form.addRow("数据集 YAML", data_row)
        config_form.addRow("训练任务", self.task_combo)
        config_form.addRow("训练模型", model_row)
        config_form.addRow("自定义模型", custom_row)
        config_form.addRow(self.model_hint)
        config_form.addRow("轮数 epochs", self.epochs_spin)
        config_form.addRow("图片尺寸 imgsz", self.imgsz_spin)
        config_form.addRow("批大小 batch", self.batch_spin)
        config_form.addRow("设备", self.device_combo)
        config_form.addRow("数据线程 workers", self.workers_spin)
        config_form.addRow("早停耐心 patience", self.patience_spin)
        config_form.addRow("初始学习率 lr0", self.lr0_spin)
        config_form.addRow("优化器", self.optimizer_combo)

        self.status_label = QLabel("尚未开始训练")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setStyleSheet("background: #202226; color: #ddd;")

        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_view, 1)

        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button = QPushButton("终止训练")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        footer = QHBoxLayout()
        footer.addWidget(self.start_button)
        footer.addWidget(self.stop_button)
        footer.addStretch(1)
        footer.addWidget(close_button)

        layout = QVBoxLayout(self)
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setWidget(config_group)
        layout.addWidget(config_scroll, 3)
        layout.addWidget(progress_group, 2)
        layout.addLayout(footer)
        layout.addWidget(
            QLabel(
                "提示：关闭窗口不会终止训练，可从托盘菜单重新打开查看进度。"
            )
        )

    def _update_model_controls(self) -> None:
        custom = self.model_combo.currentData() == "custom"
        self.custom_model_edit.setEnabled(custom)
        self.custom_model_button.setEnabled(custom)
        self.scale_combo.setEnabled(not custom)

    def _browse_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择训练模型", self.custom_model_edit.text(),
            "YOLO 模型 (*.pt *.yaml *.yml)",
        )
        if selected:
            self.custom_model_edit.setText(selected)

    def current_config(self) -> dict[str, Any]:
        data_yaml = self.data_edit.text().strip()
        if not data_yaml:
            raise ValueError("请先选择 YOLO 数据集配置文件。")
        if self.model_combo.currentData() == "custom":
            model = Path(self.custom_model_edit.text().strip())
            if not model.is_file() or model.suffix.lower() not in (".pt", ".yaml", ".yml"):
                raise ValueError("请选择有效的 .pt 权重或 .yaml 模型架构。")
        device = self.device_combo.currentData()
        return {
            "data_yaml": data_yaml,
            "model_family": self.model_combo.currentData(),
            "model_scale": self.scale_combo.currentData(),
            "custom_model": self.custom_model_edit.text().strip(),
            "task": self.task_combo.currentData(),
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "lr0": self.lr0_spin.value(),
            "device": None if device == "auto" else device,
            "optimizer": self.optimizer_combo.currentData(),
        }

    def set_dataset_path(self, path: str) -> None:
        self.data_edit.setText(path)

    def set_training_started(self, task: str, summary: str) -> None:
        self._running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(summary)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self._append_log(f"[开始训练] {task}：{summary}")

    def update_progress(self, epoch: int, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(epoch, total))
        self.status_label.setText(f"训练中… 第 {epoch}/{total} 轮")

    def append_log(self, message: str) -> None:
        self._append_log(message)

    def set_training_finished(self, message: str, error: bool = False) -> None:
        self._running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if not error else 0)
        self.status_label.setText(message)
        self._append_log(f"[{'失败' if error else '完成'}] {message}")

    def set_training_stopped(self, message: str) -> None:
        self._running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)
        self._append_log(f"[终止] {message}")

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_start(self) -> None:
        try:
            config = self.current_config()
        except ValueError as error:
            QMessageBox.warning(self, "训练配置", str(error))
            return
        self.start_requested.emit(config)

    def _on_stop(self) -> None:
        self.status_label.setText("正在终止训练…")
        self.stop_button.setEnabled(False)
        self.stop_requested.emit()

    def _browse_data(self) -> None:
        initial = self.data_edit.text().strip() or str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 数据集配置",
            initial,
            "YOLO 数据集配置 (*.yaml *.yml)",
        )
        if selected:
            self.data_edit.setText(selected)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()
