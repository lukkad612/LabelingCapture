# LabelingCapture

用于截图、摄像机采集和已有数据集标注的桌面工具。支持 YOLO 检测、姿态关键点、实例分割和旋转框标签，以及模型预标注、训练与 ONNX 导出。

[仓库](https://github.com/lukkad612/LabelingCapture) · [版本与更新](https://github.com/lukkad612/LabelingCapture/tags)

## 功能

- 导入数据集、图片、视频和对应标签；视频可按指定帧间隔提取图片。
- 修改已有数据集后保存回源标签，保留原文件名；关闭或切换数据集时提示处理未保存的修改。
- 检测框支持移动、缩放、旋转；旋转后仍可拖动角点调整大小，角度滑块用于粗调，数值框用于微调。
- 姿态关键点与四点透视预览；分割支持魔棒、画笔和橡皮。
- 标注列表选择与删除、撤销和重做、图片翻页、缩放和平移、辅助十字和放大定位。
- 摄像机截帧、实时推理、MP4 录制和 OBS 虚拟摄像机画面比例设置。
- 训练模型默认 YOLO26，可选择 YOLO11、YOLOv8、模型规模或自定义 PT / YAML。
- 可配置全局截图、保存、撤销重做、删除、确认框、关键点、方向微调、类别、翻页、分割工具和摄像机截帧快捷键。
- 主窗口支持调整大小和标题栏最大化；使用引导中提供功能说明和仓库版本链接。

## 运行源码

本次发布在 Windows 上验证。需要 Python 3.10 或更高版本，以及带系统托盘的图形桌面。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

模型预标注、训练和 ONNX 导出需另外安装模型依赖：

```powershell
python -m pip install -r requirements-model.txt
```

首次启动会在程序目录生成本地配置 `config.json`，可参考 `config.example.json`。请将项目放在具有写入权限的位置。启动后显示标注工作台；关闭工作台后仍可通过系统托盘图标重新打开，彻底退出请使用托盘菜单的“退出”。

`models/` 中保留当前项目的三份默认 YOLO11n 检测、姿态和分割权重。它们用于预标注；训练窗口中的模型选择与预标注模型独立。首次使用未缓存的预训练权重时需要联网下载，也可选择本地 PT 权重或 YAML 架构。自定义模型需与所选任务匹配。

## 常用操作

1. 选择标注任务，然后通过“导入数据集”“导入图片”“导入视频”或“截图”准备素材。
2. 检测模式下点击两个对角位置画框；右键或默认 Enter 可确认预览框。选中后拖动框内移动，拖动角点调整大小。
3. 姿态模式先画框并选中目标，再切换关键点工具；分割模式使用魔棒、画笔与橡皮编辑目标区域。
4. 默认 Ctrl+P 进行模型预标注，人工检查后用 Ctrl+S 保存。
5. 默认 A / D 翻页，Delete 删除选中框，Ctrl+Z / Ctrl+Y 撤销和重做。
6. 在“设置”的快捷键栏目修改按键。方向键默认微调 1 像素，Shift+方向键微调 10 像素，数字 1–9 切换目标类别。
7. Ctrl+滚轮缩放，滚轮纵向平移，Shift+滚轮横向平移，双击中键恢复视图。
8. 使用 OBS 虚拟摄像机时，先选择自动画面比例；若画面变形，再选择与 OBS 输出一致的比例。

类别列表顺序对应 YOLO 类别 ID。为已有数据集标注时，请保持类别含义一致。

## 保存与数据格式

导入的已有标签直接保存回源文件。程序会检查源标签是否被外部修改，发生冲突时提示重新导入。

新截图和采集样本按 `sample_000001`、`sample_000002` 等编号保存，编号自动避开已有文件。默认数据根目录为 `datasets/`，检测样本结构如下：

```text
datasets/
└── detect/
    └── datasets1/
        ├── images/sample_000001.png
        ├── labels/sample_000001.txt
        └── meta/sample_000001.json
```

姿态目录使用 `pose/`，分割目录使用 `seg/`。带旋转角度的检测框保存至 `detect/datasets_obb/`，标签为 YOLO OBB 四角点格式。视频提帧另建 `video_000001/images/` 等编号目录。

| 标签类型 | 每个目标一行 |
| --- | --- |
| Detect | `class_id cx cy width height` |
| Pose | 检测框字段后追加各关键点的 `x y visibility` |
| Segment | `class_id x1 y1 x2 y2 ...` |
| OBB | `class_id x1 y1 x2 y2 x3 y3 x4 y4` |

标签坐标按原图宽高归一化。JSON 元数据保留像素坐标、类别和透视参数；透视预览不改写原图坐标。

## 模型训练

在“训练”窗口选择数据集 YAML、任务、模型系列和规模，或指定自定义模型；调整 epochs、imgsz、batch、设备、workers、学习率等参数后开始训练。

训练在后台运行。关闭训练窗口只是隐藏，可从工作台或托盘菜单重新打开；终止按钮请求停止训练。训练成功后，将最佳权重保存为 `models/<任务>/train_000001.pt` 等编号文件，并切换该任务的预标注模型。“最多保存模型数”控制保留数量。

从“导出 ONNX”选择保存位置进行 PT 模型转换。

## 测试

在项目根目录运行：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
```

2026-11-21 版本通过 38 项测试，并用小型测试数据完成 YOLO26 单轮 CPU 训练。发布包已验证能够启动，并保留标题栏最小化、最大化和关闭按钮。

## Windows 打包

先安装基础与模型依赖，在项目根目录执行：

```powershell
python -m PyInstaller --noconfirm labeling.spec
```

产物位于 `dist/LabelingCapture/`，分发时需保留整个目录。

生成安装包前，先准备安装载荷中的源码与许可：

```powershell
$payload = (Resolve-Path ".\dist\LabelingCapture").Path
$snapshot = Join-Path $payload "Source"
New-Item -ItemType Directory -Force -Path $snapshot | Out-Null

Get-ChildItem -File | Where-Object {
    $_.Extension -in ".py", ".spec" -or
    $_.Name -like "requirements*.txt" -or
    $_.Name -in "config.example.json", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"
} | Copy-Item -Destination $snapshot -Force

foreach ($folder in "picture", "models", "tests", "installer", "third_party_licenses") {
    Copy-Item -LiteralPath $folder -Destination $snapshot -Recurse -Force
}
Copy-Item -LiteralPath LICENSE, THIRD_PARTY_NOTICES.md, README.md -Destination $payload -Force
```

安装 Inno Setup 6，将编译器加入 PATH，或使用编译器完整路径，然后执行：

```powershell
ISCC.exe "/DPayloadDir=$payload" "/O$PWD\outputs" .\installer\LabelingCapture.iss
```

安装向导支持选择安装目录、是否创建桌面和开始菜单快捷方式，并提供卸载程序。默认安装至当前用户的本地程序目录。

## 项目结构

```text
main.py                    应用控制器、工作台和导入流程
annotation.py              画布与标注编辑
capture_dialog.py          摄像机监视、截帧和录制
geometry.py                几何与透视变换
storage.py / naming.py     标签读写与输出编号
modeling.py                推理、训练和模型导出
training_dialog.py         训练配置与进度
settings_dialog.py         设置界面
shortcuts.py               快捷键默认值
guide_dialog.py            使用引导
config.example.json        示例配置
picture/                   图标与引导图片
models/                    默认预标注权重
tests/                     自动化测试
labeling.spec              PyInstaller 配置
installer/                 Inno Setup 配置与中文语言文件
third_party_licenses/      随源码保留的第三方许可文本
```

## 许可与素材

自研源码许可见 [LICENSE](LICENSE)，第三方组件与素材说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Ultralytics 的许可原文保留在 [third_party_licenses/Ultralytics-LICENSE](third_party_licenses/Ultralytics-LICENSE)。第三方权重、组件和引导图片保留各自的许可与权利归属说明。
