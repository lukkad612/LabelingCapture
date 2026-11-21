from __future__ import annotations

import logging
import re
from naming import numbered_output
import shutil
from tempfile import TemporaryDirectory
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


class TrainingStopped(Exception):
    """训练被用户提前终止。"""


class _CallbackLogHandler(logging.Handler):
    """将 Ultralytics 日志转发到 GUI 回调。"""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(self.format(record))
        except Exception:
            pass


def _import_yolo(operation: str = "运行"):
    missing_message = (
        f"{operation}依赖未安装。请在 Python 环境中安装 ultralytics 和 PyTorch 后重启。"
    )
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        if error.name == "ultralytics.utils.callbacks.clearml":
            raise RuntimeError(
                "当前 Ultralytics 安装缺少 ClearML 回调模块，安装可能不完整。"
                "请在运行软件的 Python 环境中执行："
                "python -m pip install --upgrade --force-reinstall ultralytics"
            ) from error
        raise RuntimeError(missing_message) from error
    except ImportError as error:
        raise RuntimeError(missing_message) from error
    return YOLO


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predict_annotations(
    image_rgb: np.ndarray,
    model_path: Path,
    classes: list[str],
    task: str,
    *,
    model=None,
) -> list[dict[str, Any]]:
    if model is None:
        model = load_prediction_model(model_path)

    expected_task = task
    model_task = getattr(model, "task", None)
    if model_task != expected_task:
        raise ValueError(
            f"当前任务为 {task}，但权重任务为 {model_task or '未知'}；请选择匹配的模型。"
        )

    return _predict_with_model(image_rgb, model, classes, task)


def load_prediction_model(model_path: Path):
    YOLO = _import_yolo("模型运行")

    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{model_path}")
    return YOLO(str(model_path))


def _predict_with_model(image_rgb: np.ndarray, model, classes: list[str], task: str) -> list[dict[str, Any]]:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(source=image_bgr, verbose=False)
    if not results:
        return []
    result = results[0]
    names = getattr(result, "names", getattr(model, "names", {}))
    name_to_class = {name: index for index, name in enumerate(classes)}

    def class_id_for(index: int) -> int:
        name = (
            names.get(index, str(index))
            if isinstance(names, dict)
            else names[index] if index < len(names) else str(index)
        )
        if len(classes) == 1:
            return 0
        if name not in name_to_class:
            raise ValueError(f"模型类别“{name}”不在当前类别列表中，请先调整类别设置。")
        return name_to_class[name]

    annotations: list[dict[str, Any]] = []
    if task == "segment":
        if result.masks is None:
            return annotations
        boxes = result.boxes
        for index, polygon in enumerate(result.masks.xy):
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
            if len(points) < 3:
                continue
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            class_index = int(_as_numpy(boxes.cls[index]).item())
            annotations.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "class_id": class_id_for(class_index),
                    "keypoints": [],
                    "polygon": [tuple(point) for point in points],
                }
            )
        return annotations

    if result.boxes is None:
        return annotations
    boxes = _as_numpy(result.boxes.xyxy)
    class_indices = _as_numpy(result.boxes.cls).astype(int)
    keypoints = (
        _as_numpy(result.keypoints.xy)
        if task == "pose" and result.keypoints is not None
        else None
    )
    confidences = (
        _as_numpy(result.keypoints.conf)
        if (
            task == "pose"
            and result.keypoints is not None
            and result.keypoints.conf is not None
        )
        else None
    )
    if task == "pose" and keypoints is None:
        raise ValueError("所选 pose 权重没有返回关键点，请检查模型文件。")
    for index, box in enumerate(boxes):
        points: list[tuple[float, float] | None] = []
        if keypoints is not None:
            for point_index, point in enumerate(keypoints[index]):
                if confidences is not None and confidences[index][point_index] < 0.25:
                    points.append(None)
                else:
                    points.append((float(point[0]), float(point[1])))
        annotations.append(
            {
                "box": [float(value) for value in box],
                "class_id": class_id_for(int(class_indices[index])),
                "keypoints": points,
            }
        )
    return annotations


def export_onnx(source: Path, destination: Path) -> None:
    import onnx

    with TemporaryDirectory(prefix=".onnx-", dir=destination.parent) as temporary:
        weights = Path(temporary) / source.name
        shutil.copy2(source, weights)
        exported = _import_yolo("ONNX 导出")(str(weights)).export(
            format="onnx", dynamic=True, simplify=False, device="cpu"
        )
        onnx.checker.check_model(str(exported))
        Path(exported).replace(destination)


def resolve_dataset_config(data_yaml: Path) -> dict:
    from ultralytics.utils import YAML

    data_yaml = data_yaml.resolve()
    data = YAML.load(data_yaml)
    local_root = data_yaml.parent
    configured = (local_root / (data.get("path") or ".")).resolve()

    def resolve_entry(entry: str, root: Path) -> Path:
        original = Path(entry)
        path = (root / original).resolve()
        if path.exists():
            return path
        if original.is_absolute() and root == local_root and original.is_relative_to(configured):
            return (root / original.relative_to(configured)).resolve()
        return path

    for root in dict.fromkeys((configured, local_root)):
        resolved = {}
        missing = []
        for split in ("train", "val", "test", "minival"):
            value = data.get(split)
            if not value:
                continue
            paths = [resolve_entry(entry, root) for entry in (value if isinstance(value, list) else [value])]
            missing.extend(path for path in paths if not path.exists())
            resolved[split] = [str(path) for path in paths] if isinstance(value, list) else str(paths[0])
        if not missing:
            return {**data, **resolved, "path": str(root)}
    raise FileNotFoundError(f"数据集 {data_yaml.name} 找不到图片目录或列表：{missing[0]}")


def training_model_path(family: str, scale: str, task: str, custom: str = "") -> Path:
    if task not in ("detect", "pose", "segment", "obb"):
        raise ValueError("不支持的训练任务")
    if family == "custom":
        path = Path(custom)
        if not path.is_file() or path.suffix.lower() not in (".pt", ".yaml", ".yml"):
            raise ValueError("请选择有效的自定义权重或模型架构。")
        return path.resolve()
    if family not in ("yolo26", "yolo11", "yolov8") or scale not in "n s m l x".split():
        raise ValueError("不支持的训练模型或规模")
    suffix = {"detect": "", "pose": "-pose", "segment": "-seg", "obb": "-obb"}[task]
    return Path(f"{family}{scale}{suffix}.pt")


def train_model(
    data_yaml: Path,
    model_path: Path,
    model_dir: Path,
    max_models: int,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    workers: int = 8,
    patience: int = 100,
    lr0: float | None = None,
    optimizer: str = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    expected_task: str | None = None,
) -> Path:
    YOLO = _import_yolo("训练")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"找不到数据集配置：{data_yaml}")
    builtin_obb = model_path == Path("yolo11n-obb.yaml")
    if not model_path.is_file() and not builtin_obb:
        if model_path.parent != Path(".") or not re.fullmatch(
            r"(?:yolo26|yolo11|yolov8)[nsm lx](?:-pose|-seg|-obb)?\.pt".replace(" ", ""),
            model_path.name,
        ):
            raise FileNotFoundError(f"找不到初始模型权重：{model_path}")
        from ultralytics.utils.downloads import attempt_download_asset
        cache = model_dir.parent / "pretrained"
        cache.mkdir(parents=True, exist_ok=True)
        model_path = cache / model_path.name
        if not model_path.exists():
            if log_callback:
                log_callback(f"正在下载训练权重：{model_path.name}")
            attempt_download_asset(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"训练权重下载失败，请联网重试或选择本地模型：{model_path}")
    if max_models < 1:
        raise ValueError("最多保存模型数必须至少为 1")
    if epochs < 1:
        raise ValueError("训练轮数必须至少为 1")
    if imgsz < 32:
        raise ValueError("图片尺寸必须至少为 32")
    if batch < 1:
        raise ValueError("批大小必须至少为 1")

    # Reserve an empty run directory before loading so simultaneous jobs cannot collide.
    with numbered_output(model_dir, "train_", ("", "runs")) as run_name:
        (model_dir / "runs" / run_name).mkdir(parents=True, exist_ok=False)
    model = YOLO(str(model_path))
    if expected_task is not None and model.task != expected_task:
        raise ValueError(f"模型任务为 {model.task}，与选择的 {expected_task} 不一致。")

    if progress_callback is not None:
        def _on_train_epoch_end(trainer) -> None:
            progress_callback(
                int(getattr(trainer, "epoch", 0)) + 1,
                int(getattr(trainer, "epochs", 0)),
            )

        model.add_callback("on_train_epoch_end", _on_train_epoch_end)

    if should_stop is not None:
        def _on_train_batch_end(trainer) -> None:
            if should_stop():
                trainer.stop = True

        model.add_callback("on_train_batch_end", _on_train_batch_end)

    handler = None
    if log_callback is not None:
        from ultralytics.utils import LOGGER

        handler = _CallbackLogHandler(log_callback)
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)

    try:
        train_kwargs: dict[str, Any] = {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "patience": patience,
            "optimizer": optimizer,
            "project": str(model_dir / "runs"),
            "name": run_name,
            "exist_ok": True,
        }
        if device is not None:
            train_kwargs["device"] = device
        if lr0 is not None:
            train_kwargs["lr0"] = lr0
        from ultralytics.utils import YAML

        with TemporaryDirectory(prefix="labeling-train-") as temporary:
            resolved_yaml = Path(temporary) / data_yaml.name
            YAML.save(resolved_yaml, resolve_dataset_config(data_yaml))
            model.train(data=str(resolved_yaml), **train_kwargs)
    finally:
        if handler is not None:
            from ultralytics.utils import LOGGER

            LOGGER.removeHandler(handler)

    if should_stop is not None and should_stop():
        raise TrainingStopped()

    save_dir = Path(getattr(model.trainer, "save_dir", ""))
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"训练结束但未生成 best.pt：{best_model}")

    model_dir.mkdir(parents=True, exist_ok=True)
    saved_model = model_dir / f"{run_name}.pt"
    shutil.copy2(best_model, saved_model)
    return saved_model


def prune_saved_models(model_dir: Path, max_models: int, active_model: Path) -> None:
    if max_models < 1:
        raise ValueError("最多保存模型数必须至少为 1")
    model_dir = model_dir.resolve()
    active_model = active_model.resolve()
    managed_models = sorted(
        model_dir.glob("train_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    retained = [active_model] + [
        path.resolve() for path in managed_models if path.resolve() != active_model
    ][: max_models - 1]
    runs_dir = (model_dir / "runs").resolve()
    for obsolete in managed_models:
        if obsolete.resolve() in retained:
            continue
        obsolete.unlink()
        run_dir = (runs_dir / obsolete.stem).resolve()
        if run_dir.parent != runs_dir:
            raise ValueError(f"拒绝清理模型目录之外的路径：{run_dir}")
        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir)
