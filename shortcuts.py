"""Shared defaults for settings, canvas and dialogs."""

SHORTCUTS = {
    "save": ("保存标注", "Ctrl+S"),
    "toggle_task": ("切换标注模式", "Ctrl+T"),
    "prelabel": ("模型预标注", "Ctrl+P"),
    "cancel": ("取消 / 关闭标注", "Esc"),
    "previous_image": ("上一张图片", "A"),
    "next_image": ("下一张图片", "D"),
    "magic_wand": ("分割魔棒", "M"),
    "brush": ("分割画笔", "B"),
    "eraser": ("分割橡皮", "E"),
    "undo": ("撤销", "Ctrl+Z"),
    "redo": ("重做", "Ctrl+Y"),
    "delete": ("删除选中框", "Delete"),
    "confirm_box": ("确认正在绘制的框", "Return"),
    "clear_point": ("清除选中关键点", "Backspace"),
    "camera_capture": ("摄像机截帧", "Space"),
}
for name, label, key in (("left", "左", "Left"), ("right", "右", "Right"),
                         ("up", "上", "Up"), ("down", "下", "Down")):
    SHORTCUTS[f"move_{name}"] = (f"向{label}微调 1 像素", key)
    SHORTCUTS[f"move_{name}_fast"] = (f"向{label}微调 10 像素", f"Shift+{key}")
for index in range(1, 10):
    SHORTCUTS[f"class_{index}"] = (f"切换至第 {index} 个类别", str(index))

DEFAULT_SHORTCUTS = {name: value[1] for name, value in SHORTCUTS.items()}
