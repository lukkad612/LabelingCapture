from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


GUIDE_VERSION = 3
GUIDE_NAME = "小蓝"

SCENES = {
    "welcome": (
        "标注工作台",
        f"嗨，我是{GUIDE_NAME}。选一项想了解的操作，随时可以回到这里换个主题。",
        (
            ("整理已有数据集", "dataset"),
            ("截图、摄像机和视频", "capture"),
            ("画框、姿态和实例分割", "edit"),
            ("保存、撤销与复查", "save"),
            ("模型预标注、训练和导出", "model"),
            ("快捷键与工作台", "workspace"),
        ),
    ),
    "dataset": (
        "打开已有数据集",
        "先选检测、姿态或分割任务，再点「导入数据集」。工作台会读取 images 和对应的 labels。"
        "标签另放时用「批量标注」，也可以用「单个标注」指定当前图的标签文件。",
        (("图片怎么翻页？", "navigation"), ("回到目录", "welcome")),
    ),
    "navigation": (
        "浏览和放大图片",
        "用 A / D 或预览旁的按钮翻页。上一张和下一张的缩略图可以帮你确认位置。"
        "在画布上用 Ctrl+滚轮缩放、滚轮上下平移、Shift+滚轮左右平移，双击鼠标中键可恢复初始视图。",
        (("继续看保存和复查", "save"), ("回到目录", "welcome")),
    ),
    "capture": (
        "准备新素材",
        "用「截图」圈出屏幕的一部分，也可以用设置中的全局截图快捷键。"
        "用「导入图片」打开本地图片。用「导入视频」时默认每隔 15 帧提取一张，间隔可自行设置。",
        (("看看摄像机怎么用", "camera"), ("回到目录", "welcome")),
    ),
    "camera": (
        "摄像机与 OBS 虚拟摄像机",
        "选择摄像机后可截帧标注或录制 MP4。启用「实时推理」后，截帧会带上当前预览的检测结果。"
        "使用 OBS 时先试「自动」比例，画面变形时可在「画面比例」里选与 OBS 输出一致的比例。",
        (("看看怎么画框", "edit"), ("回到目录", "welcome")),
    ),
    "edit": (
        "给目标画框",
        "检测模式下，左键定起点、右键确认框。拖动框内可移动，拖动选中框的角点可改大小。"
        "右侧角度滑块可以粗调，旁边的数值框可以每次微调 0.1°。",
        (("姿态关键点", "pose"), ("实例分割", "segment"), ("保存和复查", "save"), ("回到目录", "welcome")),
    ),
    "pose": (
        "标注姿态与透视",
        "切到 pose 后，先画框并选中目标，再用「关键点」依次点击需要的位置。"
        "右侧可调整透视点和输出尺寸，并查看选中目标的矫正预览。",
        (("继续看实例分割", "segment"), ("回到目录", "welcome")),
    ),
    "segment": (
        "用魔棒、画笔和橡皮做分割",
        "切到 segment 后，用魔棒按颜色取区域，再用画笔补选、橡皮擦除。"
        "画笔与橡皮共用半径设置，默认用 M、B、E 切换这三种工具。",
        (("继续看保存和复查", "save"), ("回到目录", "welcome")),
    ),
    "save": (
        "保存、撤销与复查",
        "Ctrl+S 保存，底部显示标签路径和保存状态。导入的数据集会写回源标签。"
        "新素材按 sample_000001 编号保存，导入的标签保留原名。选中列表中的框，默认 Delete 删除、Ctrl+Z 撤销、Ctrl+Y 重做；快捷键可在设置中修改。"
        "翻页时会保留草稿，关闭或切换数据集时可选择保存、放弃或取消。",
        (("继续看模型功能", "model"), ("回到目录", "welcome")),
    ),
    "model": (
        "模型预标注",
        "用「导入模型」选择与当前任务匹配的模型。点「模型预标注」，或用默认快捷键 Ctrl+P，"
        "把预测结果放到当前图片上，再人工调整和保存。",
        (("训练与导出", "training"), ("回到目录", "welcome")),
    ),
    "training": (
        "训练与导出模型",
        "在「训练」窗口选择任务、YOLO 的 data.yaml 和训练模型，默认 YOLO26。可选规模或自定义 PT / YAML。预训练权重首次使用需要联网下载。训练可以在后台运行。"
        "如果要把 PT 模型转换为 ONNX，用「导出 ONNX」并选择保存位置。",
        (("回到目录", "welcome"), ("进入工作台", None)),
    ),
    "workspace": (
        "设置快捷键与窗口大小",
        "在「设置」里可以改保存、撤销重做、删除、确认框、关键点、方向微调、类别、翻页、分割工具、摄像机截帧和全局截图快捷键。"
        "想扩大工作区，可拖动窗口边缘，或点击标题栏的最大化。"
        "关闭工作台后，可以点击系统托盘图标重新打开。",
        (("回到目录", "welcome"), ("进入工作台", None)),
    ),
}

class GuideDialog(QDialog):
    def __init__(self, artwork: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{GUIDE_NAME} · 标注工作台入门")
        self.setMinimumWidth(570)
        self.setStyleSheet("""
            GuideDialog { background: #111e39; }
            QLabel { color: #e8f2ff; }
            QLabel#sceneTitle { color: #8bc6ff; font-size: 22px; font-weight: bold; }
            QLabel#speaker { color: #a9d5ff; font-weight: bold; }
            QLabel#dialogue { font-size: 15px; background: #192d50; padding: 14px; border-radius: 6px; }
            QPushButton { color: #e9f3ff; background: #254776; border: 1px solid #476b9d; padding: 8px 12px; }
            QPushButton:hover, QPushButton:focus { background: #3464a0; }
        """)
        self.artwork = QPixmap(str(artwork))
        header = QHBoxLayout()
        self.portrait = QLabel()
        self.portrait.setFixedSize(108, 108)
        self.portrait.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.portrait)
        identity = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("sceneTitle")
        identity.addWidget(self.title)
        speaker = QLabel(GUIDE_NAME)
        speaker.setObjectName("speaker")
        identity.addWidget(speaker)
        header.addLayout(identity, 1)

        self.dialogue = QLabel()
        self.dialogue.setObjectName("dialogue")
        self.dialogue.setWordWrap(True)
        self.dialogue.setMinimumHeight(112)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.dialogue)
        choices = QGridLayout()
        self.buttons = [QPushButton() for _ in range(max(len(options) for _, _, options in SCENES.values()))]
        for index, button in enumerate(self.buttons):
            button.clicked.connect(lambda _checked=False, index=index: self.choose(index))
            choices.addWidget(button, index // 2, index % 2)
        layout.addLayout(choices)
        repository = QLabel(
            '<a href="https://github.com/lukkad612/LabelingCapture/tags" '
            'style="color:#8bc6ff">LabelingCapture 仓库 · 版本与更新</a>'
        )
        repository.setOpenExternalLinks(True)
        repository.setToolTip("https://github.com/lukkad612/LabelingCapture/tags")
        layout.addWidget(repository)
        skip = QPushButton("跳过引导")
        skip.clicked.connect(self.accept)
        layout.addWidget(skip, alignment=Qt.AlignmentFlag.AlignRight)
        self.show_scene("welcome")

    def show_scene(self, scene: str) -> None:
        title, dialogue, self.options = SCENES[scene]
        self.title.setText(title)
        self.dialogue.setText(dialogue)
        if not self.artwork.isNull():
            column, row = {"capture": (1, 0), "edit": (0, 1), "save": (0, 2), "model": (1, 2)}.get(scene, (0, 0))
            portrait = self.artwork.copy(1035 + column * 245, 25 + row * 244, 225, 222)
            self.portrait.setPixmap(portrait.scaled(
                self.portrait.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        for index, button in enumerate(self.buttons):
            button.setVisible(index < len(self.options))
            if index < len(self.options):
                button.setText(self.options[index][0])
        self.buttons[0].setFocus()

    def choose(self, index: int) -> None:
        destination = self.options[index][1]
        if destination is None:
            self.accept()
        else:
            self.show_scene(destination)

