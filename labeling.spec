# -*- mode: python ; coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata
import sys
import sysconfig

project_dir = Path(SPECPATH)
runtime_dll_dir = Path(sys.prefix) / "Library" / "bin"
runtime_dlls = [
    "ffi.dll",
    "libbz2.dll",
    "libcrypto-3-x64.dll",
    "libexpat.dll",
    "liblzma.dll",
    "libssl-3-x64.dll",
]
binaries = [
    (str(runtime_dll_dir / name), ".")
    for name in runtime_dlls
    if (runtime_dll_dir / name).is_file()
]
if sys.platform == "win32":
    torchvision_dir = Path(sysconfig.get_paths()["purelib"]) / "torchvision"
    binaries.extend(
        (str(path), "torchvision")
        for path in torchvision_dir.glob("*.pyd")
    )
model_sources_by_task = {
    "detect": "YOLO11n.pt",
    "pose": "yolo11n-pose.pt",
    "segment": "yolo11n-seg.pt",
}
model_sources = []
for task, model_name in model_sources_by_task.items():
    candidates = [
        project_dir / "models" / task / model_name,
        project_dir / model_name,
    ]
    model_path = next((path for path in candidates if path.is_file()), None)
    if model_path is not None:
        model_sources.append((str(model_path), str(Path("models") / task)))

icon_path = project_dir / "picture" / "图标_128x128.ico"
datas = [
    (str(project_dir / "config.example.json"), "."),
    (str(project_dir / "LICENSE"), "."),
    (str(project_dir / "THIRD_PARTY_NOTICES.md"), "."),
    (str(project_dir / "picture" / "guider.jpg"), "picture"),
]
if icon_path.is_file():
    datas.append((str(icon_path), "picture"))
datas.extend(model_sources)
for distribution in ("ultralytics", "onnx", "PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6"):
    try:
        datas.extend(copy_metadata(distribution))
    except PackageNotFoundError:
        pass  # Optional wheels depend on the installed Qt and model features.

analysis = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "ultralytics.utils.callbacks.clearml",
        "ultralytics.utils.callbacks.comet",
        "ultralytics.utils.callbacks.dvc",
        "ultralytics.utils.callbacks.mlflow",
        "ultralytics.utils.callbacks.raytune",
        "ultralytics.utils.callbacks.tensorboard",
        "ultralytics.utils.callbacks.wb",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
# Use the root MSVC runtime consistently across Qt, Shiboken and Python.
runtime_names = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
root_runtimes = {name.lower() for name, _, _ in analysis.binaries if name.lower() in runtime_names}
analysis.binaries = [
    entry for entry in analysis.binaries
    if not (
        Path(entry[0]).parent.name.lower() in {"pyside6", "shiboken6"}
        and Path(entry[0]).name.lower() in root_runtimes
    )
]

# Qt imports the unversioned ICU API supplied by Windows. The build
# environment's ICU DLL exports versioned symbols and shadows System32.
if sys.platform == "win32":
    analysis.binaries = [
        entry for entry in analysis.binaries
        if entry[0].lower() != "icuuc.dll"
    ]

exe_options = {"icon": str(icon_path)} if icon_path.is_file() else {}
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LabelingCapture",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    **exe_options,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="LabelingCapture",
)
