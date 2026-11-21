from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from geometry import (
    build_homography,
    map_box,
    map_points,
    resize_rotated_box,
    rotated_box_points,
)
from modeling import predict_annotations
from shortcuts import DEFAULT_SHORTCUTS


@dataclass
class Annotation:
    box: list[float]
    class_id: int = 0
    keypoints: list[Optional[tuple[float, float]]] = field(default_factory=list)
    polygon: list[tuple[float, float]] = field(default_factory=list)
    angle: float = 0.0
    keypoint_visibility: list[int] = field(default_factory=list)


@dataclass
class AnnotationState:
    task: str
    annotations: list[Annotation]


@dataclass
class EditHistory:
    current: AnnotationState
    saved: AnnotationState
    has_saved_label: bool = False
    sample_label_path: Path | None = None
    undo: list[AnnotationState] = field(default_factory=list)
    redo: list[AnnotationState] = field(default_factory=list)

    @property
    def dirty(self) -> bool:
        return self.current != self.saved and bool(self.current.annotations or self.saved.annotations)


def _closed_edges(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(points) < 2:
        return []
    return [
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]


def magic_wand_polygon(
    image_rgb: np.ndarray,
    seed: tuple[float, float],
    tolerance: int,
) -> list[tuple[float, float]]:
    height, width = image_rgb.shape[:2]
    x = min(max(int(round(seed[0])), 0), width - 1)
    y = min(max(int(round(seed[1])), 0), height - 1)
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flood_flags = (
        4
        | cv2.FLOODFILL_FIXED_RANGE
        | cv2.FLOODFILL_MASK_ONLY
        | (255 << 8)
    )
    cv2.floodFill(
        image_rgb.copy(),
        mask,
        (x, y),
        (0, 0, 0),
        loDiff=(tolerance, tolerance, tolerance),
        upDiff=(tolerance, tolerance, tolerance),
        flags=flood_flags,
    )
    contours, _ = cv2.findContours(
        mask[1:-1, 1:-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1:
        return []
    epsilon = max(1.0, cv2.arcLength(contour, True) * 0.002)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [(float(point[0]), float(point[1])) for point in polygon]


def mask_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(
        (contour.reshape(-1, 2) for contour in contours if cv2.contourArea(contour) > 0),
        key=cv2.contourArea, reverse=True,
    )
    if not contours:
        return []
    polygon = contours[0].tolist()
    # Double back along the join so separate contours and holes fit one YOLO polygon.
    for contour in contours[1:]:
        points = np.asarray(polygon, dtype=np.float64)
        nearest = cv2.batchDistance(points.astype(np.float32), contour.astype(np.float32),
                                    cv2.CV_32F, normType=cv2.NORM_L2, K=1)
        start = int(nearest[0].argmin())
        child_start = int(nearest[1][start, 0])
        loop = np.roll(contour, -child_start, axis=0).tolist()
        polygon[start + 1:start + 1] = loop + [loop[0], polygon[start]]
    return [tuple(point) for point in polygon]


class _MagnifierView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(260, 160)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: #26282c;")
        self._image = QImage()
        self._source_rect = QRectF()
        self._center: tuple[float, float] | None = None
        self._crosshair_color = QColor(230, 235, 240)

    def clear(self) -> None:
        self._image = QImage()
        self._source_rect = QRectF()
        self._center = None
        self.update()

    def set_content(
        self,
        image: QImage,
        center: tuple[float, float] | None,
        zoom: float,
        reference_size: tuple[int, int],
    ) -> None:
        if image.isNull() or center is None:
            self.clear()
            return
        image_width, image_height = image.width(), image.height()
        reference_width, reference_height = reference_size
        if reference_width <= 0 or reference_height <= 0:
            self.clear()
            self.hide()
            return
        fit_scale = min(
            reference_width / image_width,
            reference_height / image_height,
        )
        source_width = min(image_width, self.width() / (fit_scale * zoom))
        source_height = min(image_height, self.height() / (fit_scale * zoom))
        left = min(max(center[0] - source_width / 2, 0), image_width - source_width)
        top = min(max(center[1] - source_height / 2, 0), image_height - source_height)
        self._image = image
        self._source_rect = QRectF(left, top, source_width, source_height)
        self._center = center
        self.show()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        target_rect = QRectF(self.rect())
        if self._image.isNull() or self._source_rect.isEmpty():
            painter.fillRect(self.rect(), QColor(38, 40, 44))
            painter.setPen(QPen(QColor(245, 245, 245), 1))
            painter.drawRect(target_rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.end()
            return
        painter.drawImage(target_rect, self._image, self._source_rect)
        center_x = (
            (self._center[0] - self._source_rect.left())
            * target_rect.width()
            / self._source_rect.width()
        )
        center_y = (
            (self._center[1] - self._source_rect.top())
            * target_rect.height()
            / self._source_rect.height()
        )
        center = QPointF(center_x, center_y)
        crosshair_pen = QPen(self._crosshair_color, 1)
        crosshair_pen.setCosmetic(True)
        painter.setPen(crosshair_pen)
        painter.drawLine(0, center.y(), self.width(), center.y())
        painter.drawLine(center.x(), 0, center.x(), self.height())
        border_pen = QPen(QColor(245, 245, 245), 1)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target_rect.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.end()

    def set_crosshair_color(self, color: tuple[int, int, int]) -> None:
        self._crosshair_color = QColor(*color)
        self.update()


class AnnotationCanvas(QWidget):
    changed = Signal()
    class_selected = Signal(int)
    selection_changed = Signal(int)
    pointer_position_changed = Signal(object)
    selected_box_geometry_changed = Signal(int, object, object)
    source_points_changed = Signal()
    edit_started = Signal()
    edit_finished = Signal()
    zoom_changed = Signal(float)

    def __init__(
        self,
        image_rgb: np.ndarray,
        annotations: list[Annotation],
        source_points: np.ndarray,
        parent=None,
        crosshair_color: tuple[int, int, int] = (230, 235, 240),
        box_color: tuple[int, int, int] = (50, 220, 110),
    ) -> None:
        super().__init__(parent)
        self.image = image_rgb
        self.annotations = annotations
        self.source_points = source_points
        self.task = "detect"
        self.selected_box = -1
        self.selection_locked = False
        self.selected_keypoint: tuple[int, int] | None = None
        self.tool = "box"
        self.current_class_id = 0
        self.class_count = 1
        self._drag_kind = ""
        self._drag_start = QPointF()
        self._initial_box: list[float] = []
        self._initial_polygon: list[tuple[float, float]] = []
        self._handle_index = -1
        self._draft_box: list[float] | None = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_start = QPointF()
        self._pan_origin = QPointF()
        self._cursor_pos: QPointF | None = None
        self.crosshair_color = QColor(*crosshair_color)
        self.box_color = QColor(*box_color)
        self.magic_wand_tolerance = 20
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 160)

        self.brush_radius = 15
        self._stroke_mask = None
        self._stroke_point = None
        self.set_image(image_rgb)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image = image_rgb
        self._stroke_mask = None
        self.selected_box = -1
        self.selected_keypoint = None
        self.selection_locked = False
        self._drag_kind = ""
        self._draft_box = None
        self._cursor_pos = None
        self._zoom = 1.0
        self._pan = QPointF()
        height, width = image_rgb.shape[:2]
        self._image = QImage(
            image_rgb.data, width, height, image_rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.update()

    @property
    def task_is_pose(self) -> bool:
        return getattr(self, "task", "detect") == "pose"

    def set_task(self, task: str) -> None:
        self.task = task
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool != self.tool and self._drag_kind == "stroke":
            self._stroke_mask = None
            self._drag_kind = ""
        if tool != self.tool and self._draft_box is not None:
            self._draft_box = None
            self._drag_kind = ""
        self.tool = tool
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if tool in ("point", "magic", "brush", "eraser")
            else Qt.CursorShape.ArrowCursor
        )

    def set_class(self, class_id: int) -> None:
        self.current_class_id = class_id

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(38, 40, 44))
        painter.setClipRect(self.rect())

        image_rect = QRectF(0, 0, self._image.width(), self._image.height())
        target_rect = self._fit_rect(image_rect)
        painter.drawImage(target_rect, self._image)

        for index, annotation in enumerate(self.annotations):
            selected = index == self.selected_box
            color = self.box_color
            box_rect = self._image_rect_to_widget(annotation.box)
            painter.setPen(QPen(color, 2 if selected else 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.task == "detect" and annotation.angle:
                points = rotated_box_points(annotation.box, annotation.angle)
                painter.drawPolygon(QPolygonF([self._image_to_widget(*point) for point in points]))
            else:
                painter.drawRect(box_rect)
            painter.setPen(QPen(color, 1))
            painter.drawText(box_rect.topLeft() + QPointF(3, -4), str(annotation.class_id))

            if self.task == "segment" and len(annotation.polygon) >= 3:
                polygon = [self._image_to_widget(*point) for point in annotation.polygon]
                painter.setPen(QPen(QColor(80, 190, 255), 1.5))
                painter.setBrush(QColor(80, 190, 255, 45))
                painter.drawPolygon(QPolygonF(polygon))

            if self.task_is_pose:
                visible_points = [
                    (point_index, point)
                    for point_index, point in enumerate(annotation.keypoints)
                    if point is not None
                ]
                point_color = QColor(75, 190, 255) if selected else QColor(45, 150, 195)
                painter.setPen(QPen(point_color, 1))
                for first, second in _closed_edges(
                    [point for _, point in visible_points]
                ):
                    painter.drawLine(
                        self._image_to_widget(*first),
                        self._image_to_widget(*second),
                    )
                for point_index, point in visible_points:
                    location = self._image_to_widget(*point)
                    selected_point = self.selected_keypoint == (index, point_index)
                    marker_color = (
                        QColor(255, 120, 80)
                        if selected_point
                        else point_color
                    )
                    painter.setPen(QPen(marker_color, 2))
                    painter.setBrush(marker_color)
                    painter.drawEllipse(location, 4.5, 4.5)
                    if selected:
                        painter.drawText(location + QPointF(6, -5), str(point_index + 1))

            if selected and self.tool == "box":
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(QColor(38, 40, 44))
                for point in self._corner_handles(annotation):
                    painter.drawRect(QRectF(point.x() - 3.5, point.y() - 3.5, 7, 7))

        if self.task_is_pose and 0 <= self.selected_box < len(self.annotations):
            painter.setPen(QPen(QColor(255, 210, 80), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 210, 80))
            for index, point in enumerate(self.source_points):
                location = self._image_to_widget(float(point[0]), float(point[1]))
                painter.drawEllipse(location, 4, 4)
                painter.drawText(location + QPointF(5, -5), f"S{index + 1}")

        if self._draft_box is not None:
            painter.setPen(
                QPen(self.box_color, 1.5, Qt.PenStyle.DashLine)
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._image_rect_to_widget(self._draft_box))
        cursor_image = self._widget_to_image(self._cursor_pos) if self._cursor_pos else None
        if cursor_image is not None:
            crosshair_pen = QPen(self.crosshair_color, 1)
            crosshair_pen.setCosmetic(True)
            painter.setPen(crosshair_pen)
            image_rect = self._fit_rect(image_rect)
            painter.drawLine(
                image_rect.left(), self._cursor_pos.y(),
                image_rect.right(), self._cursor_pos.y(),
            )
            painter.drawLine(
                self._cursor_pos.x(), image_rect.top(),
                self._cursor_pos.x(), image_rect.bottom(),
            )
        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_started.emit()
        self._handle_mouse_press(event)

    def _handle_mouse_press(self, event) -> None:
        self._cursor_pos = event.position()
        self.pointer_position_changed.emit(self._widget_to_image(self._cursor_pos))
        if event.button() == Qt.MouseButton.RightButton and self._draft_box is not None:
            self._finish_draft_box(self._widget_to_image_clamped(event.position()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_kind = "pan"
            self._pan_start = event.position()
            self._pan_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus()
        position = event.position()
        image_point = self._widget_to_image(position)
        if image_point is None:
            handle = self._hit_box_handle(position) if self.tool == "box" else None
            if handle is None or not self.annotations[handle[0]].angle:
                return
            image_point = self._widget_to_image_unbounded(position)

        if self.task == "segment" and self.tool in ("brush", "eraser"):
            self._begin_segmentation_stroke(image_point)
            return

        if self.task_is_pose and 0 <= self.selected_box < len(self.annotations):
            source_handle = self._hit_source_point(position)
            if source_handle >= 0:
                self._drag_kind = "source"
                self._handle_index = source_handle
                return

        if self.tool == "point":
            keypoint_hit = self._hit_keypoint(position)
            if keypoint_hit is not None:
                previous = self.selected_box
                self.selected_box, point_index = keypoint_hit
                self.selected_keypoint = keypoint_hit
                if previous != self.selected_box:
                    self.selection_changed.emit(self.selected_box)
                    self.class_selected.emit(self.annotations[self.selected_box].class_id)
                self._drag_kind = "keypoint"
                self._handle_index = point_index
                self.changed.emit()
                self.update()
                return
            box_index = self._hit_box(image_point)
            if box_index < 0 and self.selection_locked and self._hit_other_box(image_point):
                return
            if box_index >= 0 and box_index != self.selected_box:
                self.selected_box = box_index
                self.selected_keypoint = None
                self.selection_changed.emit(box_index)
                self.class_selected.emit(self.annotations[box_index].class_id)
                self.changed.emit()
                self.update()
                return
            if self.selected_box >= 0:
                annotation = self.annotations[self.selected_box]
                annotation.keypoints.append(
                    (
                        min(image_point[0], float(self.image.shape[1] - 1)),
                        min(image_point[1], float(self.image.shape[0] - 1)),
                    )
                )
                self.selected_keypoint = (self.selected_box, len(annotation.keypoints) - 1)
                self.changed.emit()
                self.update()
            return

        if self.tool == "magic" and self.task == "segment":
            polygon = magic_wand_polygon(
                self.image, image_point, self.magic_wand_tolerance
            )
            if len(polygon) < 3:
                self.changed.emit()
                return
            polygon_array = np.asarray(polygon, dtype=np.float32)
            left, top = np.min(polygon_array, axis=0)
            right, bottom = np.max(polygon_array, axis=0)
            previous = self.selected_box
            self.annotations.append(
                Annotation(
                    [float(left), float(top), float(right), float(bottom)],
                    self.current_class_id,
                    polygon=polygon,
                )
            )
            self.selected_box = len(self.annotations) - 1
            self.selected_keypoint = None
            if previous != self.selected_box:
                self.selection_changed.emit(self.selected_box)
            self.class_selected.emit(self.current_class_id)
            self.changed.emit()
            self.update()
            return

        if self._draft_box is not None:
            self._finish_draft_box(image_point)
            return

        handle = self._hit_box_handle(position)
        if handle is not None:
            box_index, handle_index = handle
            previous = self.selected_box
            self.selected_box = box_index
            self.selected_keypoint = None
            angle = self.annotations[box_index].angle
            self._drag_kind = "resize_rotated" if self.task == "detect" and angle else "resize"
            if self._drag_kind == "resize_rotated":
                image_point = self._widget_to_image_unbounded(position)
            self._handle_index = handle_index
            self._drag_start = QPointF(image_point[0], image_point[1])
            self._initial_box = self.annotations[box_index].box.copy()
            self._initial_polygon = self.annotations[box_index].polygon.copy()
            if previous != box_index:
                self.selection_changed.emit(box_index)
            self.class_selected.emit(self.annotations[box_index].class_id)
            self.changed.emit()
            self.update()
            return

        box_index = self._hit_box(image_point)
        if box_index >= 0:
            previous = self.selected_box
            self.selected_box = box_index
            self.selected_keypoint = None
            self._drag_kind = "move"
            self._drag_start = QPointF(image_point[0], image_point[1])
            self._initial_box = self.annotations[box_index].box.copy()
            self._initial_polygon = self.annotations[box_index].polygon.copy()
            if previous != box_index:
                self.selection_changed.emit(box_index)
            self.class_selected.emit(self.annotations[box_index].class_id)
            self.changed.emit()
            self.update()
            return
        if self.selection_locked and self._hit_other_box(image_point):
            return

        previous = self.selected_box
        self.selected_box = -1
        self.selection_locked = False
        self.selected_keypoint = None
        if previous != -1:
            self.selection_changed.emit(-1)
        if self.task == "segment":
            self.changed.emit()
            self.update()
            return
        self._drag_kind = "draw"
        self._drag_start = QPointF(image_point[0], image_point[1])
        self._draft_box = [image_point[0], image_point[1], *image_point]
        self.changed.emit()
        self.update()

    def _begin_segmentation_stroke(self, point: tuple[float, float]) -> None:
        if self.tool == "eraser" and self.selected_box < 0:
            return
        self._stroke_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.selected_box >= 0:
            polygon = self.annotations[self.selected_box].polygon
            if polygon:
                cv2.fillPoly(self._stroke_mask, [np.rint(polygon).astype(np.int32)], 1)
        self._drag_kind = "stroke"
        self._stroke_point = tuple(map(round, point))
        self._paint_segmentation_stroke(point)

    def _paint_segmentation_stroke(self, point: tuple[float, float]) -> None:
        current = tuple(map(round, point))
        value = int(self.tool == "brush")
        cv2.line(self._stroke_mask, self._stroke_point, current, value, 2 * self.brush_radius)
        cv2.circle(self._stroke_mask, current, self.brush_radius, value, -1)
        self._stroke_point = current
        polygon = mask_polygon(self._stroke_mask)
        index = self.selected_box
        if polygon:
            left, top, width, height = cv2.boundingRect(self._stroke_mask)
            box = [float(left), float(top), float(left + width), float(top + height)]
            if index >= 0:
                self.annotations[index].box = box
                self.annotations[index].polygon = polygon
            else:
                self.annotations.append(Annotation(box, self.current_class_id, polygon=polygon))
                self.selected_box = len(self.annotations) - 1
                self.selection_changed.emit(self.selected_box)
        elif index >= 0:
            self.delete_selected()
        self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_kind == "stroke":
            self._paint_segmentation_stroke(self._widget_to_image_clamped(event.position()))
            return
        self._cursor_pos = event.position()
        self.pointer_position_changed.emit(self._widget_to_image(self._cursor_pos))
        if not self._drag_kind:
            if self._draft_box is not None and self.tool == "box":
                self._draft_box[2:] = self._widget_to_image_clamped(event.position())
            self.update()
            return
        if self._drag_kind == "pan":
            self._pan = self._pan_origin + (event.position() - self._pan_start)
            self._clamp_pan()
            self.update()
            return
        if self._drag_kind == "resize_rotated":
            image_point = self._widget_to_image_unbounded(event.position())
        else:
            image_point = self._widget_to_image(event.position())
        if image_point is None and self._drag_kind in (
            "draw",
            "move",
            "resize",
            "keypoint",
        ):
            image_point = self._widget_to_image_clamped(event.position())
        if image_point is None:
            return
        x, y = image_point
        height, width = self.image.shape[:2]

        if self._drag_kind == "source":
            self.source_points[self._handle_index] = (
                min(max(x, -float(width)), float(width) * 2),
                min(max(y, -float(height)), float(height) * 2),
            )
            self.source_points_changed.emit()
        elif self._drag_kind == "draw":
            start_x, start_y = self._drag_start.x(), self._drag_start.y()
            self._draft_box = [start_x, start_y, x, y]
        elif self._drag_kind == "move":
            dx = x - self._drag_start.x()
            dy = y - self._drag_start.y()
            initial = self._initial_box
            previous_box = self.annotations[self.selected_box].box.copy()
            box_width, box_height = initial[2] - initial[0], initial[3] - initial[1]
            left = min(max(initial[0] + dx, 0.0), max(0.0, width - box_width))
            top = min(max(initial[1] + dy, 0.0), max(0.0, height - box_height))
            updated_box = [
                left,
                top,
                left + box_width,
                top + box_height,
            ]
            self.annotations[self.selected_box].box = updated_box
            self.annotations[self.selected_box].polygon = [
                (point[0] + left - initial[0], point[1] + top - initial[1])
                for point in self._initial_polygon
            ]
            self.selected_box_geometry_changed.emit(
                self.selected_box, previous_box, updated_box.copy()
            )
        elif self._drag_kind == "resize_rotated":
            annotation = self.annotations[self.selected_box]
            annotation.box = resize_rotated_box(
                self._initial_box, annotation.angle, self._handle_index,
                x - self._drag_start.x(), y - self._drag_start.y(), width, height,
            )
        elif self._drag_kind == "resize":
            box = self._initial_box.copy()
            previous_box = self.annotations[self.selected_box].box.copy()
            handle = self._handle_index
            if handle in (0, 3, 7):
                box[0] = x
            if handle in (1, 2, 5):
                box[2] = x
            if handle in (0, 1, 4):
                box[1] = y
            if handle in (2, 3, 6):
                box[3] = y
            left, right = sorted((min(max(box[0], 0.0), width), min(max(box[2], 0.0), width)))
            top, bottom = sorted((min(max(box[1], 0.0), height), min(max(box[3], 0.0), height)))
            updated_box = [left, top, right, bottom]
            self.annotations[self.selected_box].box = updated_box
            initial_width = self._initial_box[2] - self._initial_box[0]
            initial_height = self._initial_box[3] - self._initial_box[1]
            if initial_width > 0 and initial_height > 0:
                self.annotations[self.selected_box].polygon = [
                    (
                        left + (point[0] - self._initial_box[0]) * (right - left) / initial_width,
                        top + (point[1] - self._initial_box[1]) * (bottom - top) / initial_height,
                    )
                    for point in self._initial_polygon
                ]
            self.selected_box_geometry_changed.emit(
                self.selected_box, previous_box, updated_box.copy()
            )
        elif self._drag_kind == "keypoint":
            annotation = self.annotations[self.selected_box]
            point_index = self._handle_index
            if point_index < len(annotation.keypoints):
                annotation.keypoints[point_index] = (
                    min(x, float(width - 1)),
                    min(y, float(height - 1)),
                )

        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._handle_mouse_release(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_finished.emit()

    def _handle_mouse_release(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._drag_kind == "pan":
            self._drag_kind = ""
            self.setCursor(
                Qt.CursorShape.CrossCursor if self.tool == "point" else Qt.CursorShape.ArrowCursor
            )
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        self._stroke_mask = None
        self._drag_kind = ""
        self.changed.emit()
        self.update()

    def _finish_draft_box(self, image_point: tuple[float, float]) -> None:
        if self._draft_box is None:
            return
        start_x, start_y = self._draft_box[:2]
        left, right = sorted((start_x, image_point[0]))
        top, bottom = sorted((start_y, image_point[1]))
        if right - left >= 2 and bottom - top >= 2:
            previous = self.selected_box
            self.annotations.append(
                Annotation([left, top, right, bottom], self.current_class_id)
            )
            self.selected_box = len(self.annotations) - 1
            self.selected_keypoint = None
            if previous != self.selected_box:
                self.selection_changed.emit(self.selected_box)
            self.class_selected.emit(self.annotations[-1].class_id)
        self._draft_box = None
        self._drag_kind = ""
        self.changed.emit()
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor_pos = None
        self.pointer_position_changed.emit(None)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        configured = {**DEFAULT_SHORTCUTS, **getattr(self, "shortcuts", {})}
        pressed = QKeySequence(event.keyCombination())
        if event.key() == Qt.Key.Key_Enter and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            pressed = QKeySequence("Return")
        action = next((name for name, sequence in configured.items()
                       if pressed == QKeySequence(sequence)), None)
        if action == "confirm_box" and self._draft_box is not None:
            self._finish_draft_box(self._widget_to_image_clamped(self._cursor_pos or QPointF()))
            return
        if action == "delete":
            self.delete_selected()
            return
        if action == "clear_point":
            self.clear_selected_keypoint()
            return
        if action and action.startswith("move_"):
            step = 10.0 if action.endswith("_fast") else 1.0
            direction = action.split("_")[1]
            dx, dy = {"left": (-step, 0), "right": (step, 0),
                      "up": (0, -step), "down": (0, step)}[direction]
            self.nudge_selected(dx, dy)
            return
        if action and action.startswith("class_"):
            class_index = int(action.split("_")[1]) - 1
            if self.selected_box >= 0 and class_index < self.class_count:
                self.annotations[self.selected_box].class_id = class_index
                self.class_selected.emit(class_index)
                self.changed.emit()
                self.update()
            return
        super().keyPressEvent(event)

    def delete_selected(self) -> None:
        if 0 <= self.selected_box < len(self.annotations):
            self.annotations.pop(self.selected_box)
            self.selected_box = -1
            self.selection_locked = False
            self.selected_keypoint = None
            self.selection_changed.emit(-1)
        self.changed.emit()
        self.update()

    def clear_selected_keypoint(self) -> None:
        if self.selected_keypoint is None:
            return
        box_index, point_index = self.selected_keypoint
        if (
            0 <= box_index < len(self.annotations)
            and 0 <= point_index < len(self.annotations[box_index].keypoints)
        ):
            self.annotations[box_index].keypoints[point_index] = None
        self.selected_keypoint = None
        self.changed.emit()
        self.update()

    def nudge_selected(self, dx: float, dy: float) -> None:
        height, width = self.image.shape[:2]
        if self.selected_keypoint is not None:
            box_index, point_index = self.selected_keypoint
            annotation = self.annotations[box_index]
            point = annotation.keypoints[point_index]
            if point is not None:
                annotation.keypoints[point_index] = (
                    min(max(point[0] + dx, 0.0), float(width - 1)),
                    min(max(point[1] + dy, 0.0), float(height - 1)),
                )
        elif 0 <= self.selected_box < len(self.annotations):
            box = self.annotations[self.selected_box].box
            previous_box = box.copy()
            box_width, box_height = box[2] - box[0], box[3] - box[1]
            left = min(max(box[0] + dx, 0.0), max(0.0, width - box_width))
            top = min(max(box[1] + dy, 0.0), max(0.0, height - box_height))
            self.annotations[self.selected_box].box = [
                left,
                top,
                left + box_width,
                top + box_height,
            ]
            self.annotations[self.selected_box].polygon = [
                (point[0] + left - box[0], point[1] + top - box[1])
                for point in self.annotations[self.selected_box].polygon
            ]
            self.selected_box_geometry_changed.emit(
                self.selected_box,
                previous_box,
                self.annotations[self.selected_box].box.copy(),
            )
        self.changed.emit()
        self.update()

    def _fit_rect(self, rect: QRectF) -> QRectF:
        scale = min(self.width() / rect.width(), self.height() / rect.height()) * self._zoom
        width, height = rect.width() * scale, rect.height() * scale
        return QRectF(
            (self.width() - width) / 2 + self._pan.x(),
            (self.height() - height) / 2 + self._pan.y(),
            width,
            height,
        )

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if not delta:
                event.accept()
                return
            self._zoom_at(event.position(), delta)
            event.accept()
            return
        horizontal_delta = event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            horizontal_delta = horizontal_delta or delta
            delta = 0
        if not horizontal_delta and not delta:
            super().wheelEvent(event)
            return
        self._pan += QPointF(
            -horizontal_delta / 120 * 60,
            delta / 120 * 60,
        )
        self._clamp_pan()
        self.update()
        event.accept()

    def _zoom_at(self, anchor: QPointF, delta: int) -> None:
        image_point = self._widget_to_image(anchor)
        if image_point is None:
            image_height, image_width = self.image.shape[:2]
            image_point = (image_width / 2, image_height / 2)

        old_zoom = self._zoom
        self._zoom = min(8.0, max(0.5, old_zoom * (1.15 ** (delta / 120))))
        if abs(self._zoom - 1.0) < 1e-6:
            self._zoom = 1.0
            self._pan = QPointF()
        else:
            image_height, image_width = self.image.shape[:2]
            scale = min(self.width() / image_width, self.height() / image_height) * self._zoom
            self._pan = QPointF(
                anchor.x() - self.width() / 2 - (image_point[0] - image_width / 2) * scale,
                anchor.y() - self.height() / 2 - (image_point[1] - image_height / 2) * scale,
            )
            self._clamp_pan()
        if self._zoom != old_zoom:
            self.zoom_changed.emit(self._zoom)
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._zoom = 1.0
            self._pan = QPointF()
            self.zoom_changed.emit(self._zoom)
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _clamp_pan(self) -> None:
        image_height, image_width = self.image.shape[:2]
        scale = min(self.width() / image_width, self.height() / image_height) * self._zoom
        scaled_width, scaled_height = image_width * scale, image_height * scale
        maximum_x = abs(scaled_width - self.width()) / 2
        maximum_y = abs(scaled_height - self.height()) / 2
        self._pan.setX(min(maximum_x, max(-maximum_x, self._pan.x())))
        self._pan.setY(min(maximum_y, max(-maximum_y, self._pan.y())))

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        image_height, image_width = self.image.shape[:2]
        target = self._fit_rect(QRectF(0, 0, image_width, image_height))
        return QPointF(
            target.left() + x * target.width() / image_width,
            target.top() + y * target.height() / image_height,
        )

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        image_height, image_width = self.image.shape[:2]
        target = self._fit_rect(QRectF(0, 0, image_width, image_height))
        if not target.contains(point):
            return None
        x = (point.x() - target.left()) * image_width / target.width()
        y = (point.y() - target.top()) * image_height / target.height()
        return (
            min(max(x, 0.0), float(image_width)),
            min(max(y, 0.0), float(image_height)),
        )

    def _widget_to_image_unbounded(self, point: QPointF) -> tuple[float, float]:
        image_height, image_width = self.image.shape[:2]
        target = self._fit_rect(QRectF(0, 0, image_width, image_height))
        return (
            (point.x() - target.left()) * image_width / target.width(),
            (point.y() - target.top()) * image_height / target.height(),
        )

    def _widget_to_image_clamped(self, point: QPointF) -> tuple[float, float]:
        image_height, image_width = self.image.shape[:2]
        x, y = self._widget_to_image_unbounded(point)
        return (
            min(max(x, 0.0), float(image_width)),
            min(max(y, 0.0), float(image_height)),
        )

    def _image_rect_to_widget(self, box: list[float]) -> QRectF:
        first = self._image_to_widget(box[0], box[1])
        second = self._image_to_widget(box[2], box[3])
        return QRectF(first, second).normalized()

    def _corner_handles(self, annotation: Annotation) -> list[QPointF]:
        angle = annotation.angle if self.task == "detect" else 0.0
        return [self._image_to_widget(*point) for point in rotated_box_points(annotation.box, angle)]

    def _hit_box_handle(self, position: QPointF) -> tuple[int, int] | None:
        index = self.selected_box
        if not 0 <= index < len(self.annotations):
            return None
        for corner, point in enumerate(self._corner_handles(self.annotations[index])):
            if self._distance_manhattan(position, point) <= 11:
                return index, corner
        return None

    def _hit_box(self, point: tuple[float, float]) -> int:
        x, y = point
        for index in range(len(self.annotations) - 1, -1, -1):
            if self.selection_locked and index != self.selected_box:
                continue
            if self._contains(self.annotations[index], (x, y)):
                return index
        return -1

    def _hit_other_box(self, point: tuple[float, float]) -> bool:
        x, y = point
        for index, annotation in enumerate(self.annotations):
            if index == self.selected_box:
                continue
            if self._contains(annotation, (x, y)):
                return True
        return False

    def _contains(self, annotation: Annotation, point: tuple[float, float]) -> bool:
        # Keep a five-screen-pixel margin at every zoom level.
        scale = abs(self._image_to_widget(1, 0).x() - self._image_to_widget(0, 0).x())
        tolerance = 5.0 / max(scale, 1e-6)
        contour = rotated_box_points(
            annotation.box, annotation.angle if self.task == "detect" else 0
        ).astype(np.float32)
        return cv2.pointPolygonTest(contour, point, True) >= -tolerance

    def _hit_keypoint(self, position: QPointF) -> tuple[int, int] | None:
        for box_index in range(len(self.annotations) - 1, -1, -1):
            if self.selection_locked and box_index != self.selected_box:
                continue
            for point_index, point in enumerate(self.annotations[box_index].keypoints):
                if point is None:
                    continue
                location = self._image_to_widget(*point)
                if self._distance_manhattan(position, location) <= 14:
                    return box_index, point_index
        return None

    def _hit_source_point(self, position: QPointF) -> int:
        for index, point in enumerate(self.source_points):
            if (
                self._distance_manhattan(
                    position, self._image_to_widget(float(point[0]), float(point[1]))
                )
                <= 14
            ):
                return index
        return -1

    @staticmethod
    def _distance_manhattan(first: QPointF, second: QPointF) -> float:
        return abs(first.x() - second.x()) + abs(first.y() - second.y())


class _PerspectivePreview(QLabel):
    point_dragged = Signal(int, float, float)

    def __init__(self) -> None:
        super().__init__("透视预览")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._target_points = np.empty((0, 2), dtype=np.float32)
        self._target_size = (1, 1)
        self._drag_index = -1

    def set_target_geometry(self, points: np.ndarray, width: int, height: int) -> None:
        self._target_points = np.asarray(points, dtype=np.float32).copy()
        self._target_size = (max(1, width), max(1, height))

    def _pixmap_rect(self) -> QRectF:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QRectF()
        return QRectF(
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            pixmap.width(),
            pixmap.height(),
        )

    def _point_position(self, point: np.ndarray, rect: QRectF) -> QPointF:
        width, height = self._target_size
        return QPointF(
            rect.left() + float(point[0]) * rect.width() / width,
            rect.top() + float(point[1]) * rect.height() / height,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            rect = self._pixmap_rect()
            if not rect.isEmpty():
                for index, point in enumerate(self._target_points):
                    location = self._point_position(point, rect)
                    if abs(location.x() - event.position().x()) + abs(
                        location.y() - event.position().y()
                    ) <= 18:
                        self._drag_index = index
                        event.accept()
                        return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index >= 0 and event.buttons() & Qt.MouseButton.LeftButton:
            rect = self._pixmap_rect()
            if rect.isEmpty():
                return
            width, height = self._target_size
            x = min(max((event.position().x() - rect.left()) * width / rect.width(), 0), width)
            y = min(max((event.position().y() - rect.top()) * height / rect.height(), 0), height)
            self.point_dragged.emit(self._drag_index, float(x), float(y))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_index = -1
        super().mouseReleaseEvent(event)


class AnnotationDialog(QDialog):
    save_requested = Signal(dict)
    task_changed = Signal(str)
    classes_changed = Signal(object)

    def __init__(
        self,
        image_rgb: np.ndarray,
        classes: list[str],
        parent=None,
        magnifier_zoom: float = 3.0,
        crosshair_color: tuple[int, int, int] = (230, 235, 240),
        box_color: tuple[int, int, int] = (50, 220, 110),
        magic_wand_tolerance: int = 20,
        model_path: Path | dict[str, Path] | None = None,
        shortcuts: dict[str, str] | None = None,
        perspective_source_defaults: list[list[float]] | None = None,
        perspective_target_defaults: list[list[float]] | None = None,
        initial_task: str = "detect",
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowTitle("截图标注")
        self.resize(1380, 850)
        self.image = image_rgb
        self.classes = list(classes)
        height, width = image_rgb.shape[:2]
        source_defaults = np.asarray(
            perspective_source_defaults or [[0, 0], [1, 0], [1, 1], [0, 1]],
            dtype=np.float32,
        )
        target_defaults = np.asarray(
            perspective_target_defaults or [[0, 0], [1, 0], [1, 1], [0, 1]],
            dtype=np.float32,
        )
        self.source_defaults = source_defaults
        self.target_defaults = target_defaults
        self.source_points = source_defaults * np.float32([width - 1, height - 1])
        self.target_points = target_defaults * np.float32([50, 100])
        self.model_path = model_path
        self.magic_wand_tolerance = magic_wand_tolerance
        self.annotations: list[Annotation] = []
        self.task = "detect"
        self._saved = False
        self.has_image = True
        self.source_label_path: Path | None = None
        self.confirm_close = None
        self.can_delete_class = None
        self._gesture = False
        self.edit_history = EditHistory(AnnotationState(initial_task, []), AnnotationState(initial_task, []))
        self._preview_pixmap = QPixmap()
        self._preview_image = QImage()
        self._magnifier_image = QImage()
        self._cursor_image_position: tuple[float, float] | None = None
        self.magnifier_zoom = min(8.0, max(2.0, float(magnifier_zoom)))
        self.box_color = QColor(*box_color)

        self.task_combo = QComboBox()
        self.task_combo.addItem("detect", "detect")
        self.task_combo.addItem("pose", "pose")
        self.task_combo.addItem("segment", "segment")
        initial_task_index = self.task_combo.findData(initial_task)
        if initial_task_index >= 0:
            self.task_combo.setCurrentIndex(initial_task_index)
            self.task = initial_task
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentIndexChanged.connect(self._class_changed)

        self.canvas = AnnotationCanvas(
            image_rgb,
            self.annotations,
            self.source_points,
            crosshair_color=crosshair_color,
            box_color=box_color,
        )
        self.canvas.set_task(initial_task)
        self.canvas.magic_wand_tolerance = magic_wand_tolerance
        self.canvas.class_selected.connect(self._select_class)
        self.canvas.selection_changed.connect(self._selection_changed)
        self.canvas.pointer_position_changed.connect(self._pointer_position_changed)
        self.canvas.selected_box_geometry_changed.connect(self._selected_box_geometry_changed)
        self.canvas.source_points_changed.connect(self._sync_source_controls)
        self.canvas.changed.connect(self._refresh_preview)
        self.canvas.changed.connect(self._sync_annotation_list)
        self.canvas.selection_changed.connect(self._sync_annotation_list)
        self.canvas.class_count = len(self.classes)

        box_tool = QPushButton("框选（左键起点、右键确认）/ 微调")
        point_tool = QPushButton("关键点")
        magic_tool = QPushButton("魔棒")
        box_tool.setCheckable(True)
        point_tool.setCheckable(True)
        magic_tool.setCheckable(True)
        box_tool.setChecked(True)
        point_tool.setEnabled(False)
        magic_tool.setEnabled(False)
        self.box_tool_button = box_tool
        self.point_tool_button = point_tool
        self.magic_tool_button = magic_tool
        self.brush_tool_button = QPushButton("画笔")
        self.eraser_tool_button = QPushButton("橡皮")
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 250)
        self.brush_size.setValue(self.canvas.brush_radius)
        self.brush_size.setSuffix(" px")
        self.brush_size.setToolTip("画笔与橡皮半径")
        self.brush_size.valueChanged.connect(lambda value: setattr(self.canvas, "brush_radius", value))
        for tool, button in (("brush", self.brush_tool_button), ("eraser", self.eraser_tool_button)):
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, tool=tool: self._select_canvas_tool(tool))
        self.angle_edit = QDoubleSpinBox()
        self.angle_edit.setRange(-180, 180)
        self.angle_edit.setDecimals(1)
        self.angle_edit.setSingleStep(0.1)
        self.angle_edit.setKeyboardTracking(False)
        self.angle_edit.setSuffix("°")
        self.angle_edit.setToolTip("选中检测框后输入角度，或用上下方向键以 0.1° 微调")
        self.angle_edit.valueChanged.connect(self._set_angle)
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.setPageStep(15)
        self.angle_slider.setAccessibleName("旋转角度粗调")
        self.angle_slider.setToolTip("拖动滑块快速旋转，使用右侧输入框微调")
        self.angle_slider.valueChanged.connect(lambda value: self.angle_edit.setValue(value))
        point_tool.setEnabled(initial_task == "pose")
        magic_tool.setEnabled(initial_task == "segment")
        box_tool.clicked.connect(lambda: self._select_canvas_tool("box"))
        point_tool.clicked.connect(lambda: self._select_canvas_tool("point"))
        magic_tool.clicked.connect(lambda: self._select_canvas_tool("magic"))
        if initial_task == "segment":
            self._select_canvas_tool("magic")
        add_class_button = QPushButton("新增类别")
        add_class_button.clicked.connect(self._add_class)
        delete_class_button = QPushButton("删除类别")
        delete_class_button.clicked.connect(self._delete_class)
        prelabel_button = QPushButton("模型预标注 Ctrl+P")
        prelabel_button.clicked.connect(self._run_prelabel)
        delete_button = QPushButton("删除选中框")
        delete_button.clicked.connect(self.canvas.delete_selected)
        save_button = QPushButton("保存  Ctrl+S")
        save_button.clicked.connect(self._request_save)
        cancel_button = QPushButton("取消  Esc")
        cancel_button.clicked.connect(self.reject)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("任务"))
        toolbar.addWidget(self.task_combo)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("类别"))
        toolbar.addWidget(self.class_combo)
        toolbar.addWidget(add_class_button)
        toolbar.addWidget(delete_class_button)
        toolbar.addStretch()
        tools = QHBoxLayout()
        tools.addWidget(box_tool)
        tools.addWidget(point_tool)
        tools.addWidget(prelabel_button)
        tools.addWidget(delete_button)
        self.zoom_status = QLabel("100%")
        self.canvas.zoom_changed.connect(
            lambda zoom: self.zoom_status.setText(f"{zoom:.0%}")
        )
        self.zoom_status.setToolTip(
            "滚轮平移 · Shift+滚轮横移 · Ctrl+滚轮缩放"
        )
        tools.addWidget(self.zoom_status)
        tools.addStretch(1)
        self.undo_button = QPushButton("撤销")
        self.redo_button = QPushButton("重做")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        edit_actions = QHBoxLayout()
        edit_actions.addStretch(1)
        edit_actions.addWidget(self.undo_button)
        edit_actions.addWidget(self.redo_button)
        edit_actions.addWidget(save_button)
        edit_actions.addWidget(cancel_button)

        self.preview = _PerspectivePreview()
        self.preview.setMinimumSize(250, 180)
        self.preview.setStyleSheet("background: #26282c; color: #ddd;")
        self.preview.point_dragged.connect(self._target_preview_point_dragged)
        self.preview_magnifier = _MagnifierView()
        self.preview_magnifier.set_crosshair_color(crosshair_color)
        self.preview_status = QLabel()
        self.preview_status.setWordWrap(True)
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        self.annotation_list.currentRowChanged.connect(
            self._annotation_list_selection_changed
        )
        self.annotation_list.setToolTip("选中标注框后按 Delete 删除")
        delete_from_list = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.annotation_list)
        self.delete_from_list = delete_from_list
        delete_from_list.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_from_list.setAutoRepeat(False)
        delete_from_list.activated.connect(self._delete_list_annotation)
        annotation_group = QGroupBox("已标注框")
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.addWidget(self.annotation_list)

        self.source_controls: list[list[QDoubleSpinBox]] = []
        self.target_controls: list[list[QDoubleSpinBox]] = []
        perspective_group = self._make_perspective_controls()
        self.pose_widgets = [self.preview, self.preview_status, perspective_group]
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("标注列表与定位预览"))
        right_panel.addWidget(QLabel("放大定位（鼠标位置映射到透视图）"))
        right_panel.addWidget(self.preview_magnifier, 0, Qt.AlignmentFlag.AlignHCenter)
        right_panel.addWidget(annotation_group, 1)
        right_panel.addWidget(self.preview, 2)
        right_panel.addWidget(self.preview_status)
        right_panel.addWidget(perspective_group)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setWidget(right_widget)
        sidebar.setMinimumWidth(300)
        self.sidebar_content = right_widget

        content = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        self.empty_image = QLabel("导入数据集、图片或视频以开始标注")
        self.empty_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_image.hide()
        splitter.addWidget(self.empty_image)
        splitter.addWidget(sidebar)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([850, 330])
        content.addWidget(splitter)
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addLayout(tools)
        layout.addLayout(edit_actions)
        self.segment_tools = QWidget()
        segment_row = QHBoxLayout(self.segment_tools)
        segment_row.setContentsMargins(0, 0, 0, 0)
        segment_row.addWidget(QLabel("分割工具"))
        for widget in (magic_tool, self.brush_tool_button, self.eraser_tool_button, self.brush_size):
            segment_row.addWidget(widget)
        segment_row.addStretch()
        layout.addWidget(self.segment_tools)
        layout.addLayout(content, 1)
        status = QHBoxLayout()
        self.save_status = QLabel()
        self.save_path = QLabel()
        self.save_path.setWordWrap(True)
        self.save_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status.addWidget(self.save_status)
        status.addWidget(self.save_path, 1)
        layout.addLayout(status)
        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("旋转角度"))
        angle_row.addWidget(self.angle_slider, 1)
        angle_row.addWidget(self.angle_edit)
        right_panel.insertLayout(0, angle_row)
        self._sync_edit_tools()

        self._image_actions = [box_tool, point_tool, magic_tool, prelabel_button, delete_button, save_button]
        self.action_shortcut_buttons = {
            "save": save_button, "cancel": cancel_button, "prelabel": prelabel_button,
            "delete": delete_button, "undo": self.undo_button, "redo": self.redo_button,
        }
        self.action_button_titles = {
            "save": "保存", "cancel": "取消", "prelabel": "模型预标注",
            "delete": "删除框", "undo": "撤销", "redo": "重做",
        }
        self.configure_shortcuts(shortcuts)
        self.canvas.changed.connect(self.record_edit)
        self.canvas.edit_started.connect(self.begin_edit)
        self.canvas.edit_finished.connect(self.finish_edit)
        self.angle_slider.sliderPressed.connect(self.begin_edit)
        self.angle_slider.sliderReleased.connect(self.finish_edit)
        self.reset_history()
        self._refresh_preview()
        self._sync_annotation_list()

    def begin_edit(self) -> None:
        self.record_edit()
        self._gesture = True

    def finish_edit(self) -> None:
        self._gesture = False
        self.record_edit()

    def record_edit(self) -> None:
        if self._gesture or not self.has_image:
            return
        history = self.edit_history
        if self.task != history.current.task or self.annotations != history.current.annotations:
            history.undo.append(history.current)
            del history.undo[:-50]
            history.current = AnnotationState(self.task, deepcopy(self.annotations))
            history.redo.clear()
        self._update_save_status()

    def reset_history(self, has_saved_label: bool = False) -> None:
        state = AnnotationState(self.task, deepcopy(self.annotations))
        self.edit_history = EditHistory(state, state, has_saved_label)
        self._gesture = False
        self._update_save_status()

    def restore_history(self, history: EditHistory) -> None:
        self._gesture = False
        self.edit_history = history
        self._restore_edit(history.current)

    def _restore_edit(self, state: AnnotationState) -> None:
        task_changed = self.task != state.task
        self.annotations.clear()
        self.task_combo.blockSignals(True)
        self.task_combo.setCurrentIndex(self.task_combo.findData(state.task))
        self.task_combo.blockSignals(False)
        self.task = state.task
        self.canvas.set_task(state.task)
        self.annotations.extend(deepcopy(state.annotations))
        self.canvas._draft_box = None
        self.canvas._drag_kind = ""
        self.canvas._stroke_mask = None
        self.canvas.selected_keypoint = None
        self.canvas.selection_locked = False
        if self.canvas.selected_box >= len(self.annotations):
            self.canvas.selected_box = -1
        if task_changed:
            self._select_canvas_tool("magic" if self.task == "segment" else "box")
        self.point_tool_button.setEnabled(self.has_image and self.task == "pose")
        self.magic_tool_button.setEnabled(self.has_image and self.task == "segment")
        if self.canvas.selected_box >= 0:
            self._select_class(self.annotations[self.canvas.selected_box].class_id)
        self.canvas.selection_changed.emit(self.canvas.selected_box)
        self.canvas.changed.emit()
        self.canvas.update()

    def undo(self) -> None:
        if self._gesture or not self.edit_history.undo:
            return
        history = self.edit_history
        history.redo.append(history.current)
        history.current = history.undo.pop()
        self._restore_edit(history.current)

    def redo(self) -> None:
        if self._gesture or not self.edit_history.redo:
            return
        history = self.edit_history
        history.undo.append(history.current)
        history.current = history.redo.pop()
        self._restore_edit(history.current)

    def _update_save_status(self) -> None:
        history = self.edit_history
        self.undo_button.setEnabled(self.has_image and bool(history.undo))
        self.redo_button.setEnabled(self.has_image and bool(history.redo))
        if not self.has_image:
            self.save_status.clear()
            self.save_path.clear()
            return
        if history.dirty:
            self.setWindowTitle(self.windowTitle().removesuffix(" · 已保存"))
            status = "已修改（待保存为空）" if not self.annotations and self.source_label_path else "已修改（未保存）"
        elif history.has_saved_label:
            status = "已保存为空" if not self.annotations else "已保存"
        else:
            status = "未标注"
        self.save_status.setText(status)
        path = self.source_label_path or history.sample_label_path
        self.save_path.setText(str(path) if path else "保存时生成新样本")
        self.save_path.setToolTip(self.save_path.text())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def configure_shortcuts(self, shortcuts: dict[str, str] | None) -> None:
        for shortcut in getattr(self, "_shortcuts", []):
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts = []
        shortcut_defaults = DEFAULT_SHORTCUTS
        configured_shortcuts = {**shortcut_defaults, **(shortcuts or {})}
        self.canvas.shortcuts = configured_shortcuts
        self.delete_from_list.setKey(QKeySequence(configured_shortcuts["delete"]))
        self.annotation_list.setToolTip(f"选中标注框后按 {configured_shortcuts['delete']} 删除")
        for name, button in self.action_shortcut_buttons.items():
            button.setText(f"{self.action_button_titles[name]} ({configured_shortcuts[name]})")
        for name, button, title in (
            ("magic_wand", self.magic_tool_button, "魔棒"),
            ("brush", self.brush_tool_button, "画笔"),
            ("eraser", self.eraser_tool_button, "橡皮"),
        ):
            button.setText(f"{title} ({configured_shortcuts[name]})")
        for key_name, callback in (
            ("undo", self.undo),
            ("redo", self.redo),
            ("save", self._request_save),
            ("toggle_task", self._toggle_task),
            ("prelabel", self._run_prelabel),
            ("cancel", self.reject),
            ("magic_wand", lambda: self._segment_shortcut("magic")),
            ("brush", lambda: self._segment_shortcut("brush")),
            ("eraser", lambda: self._segment_shortcut("eraser")),
        ):
            shortcut = QShortcut(
                QKeySequence(configured_shortcuts.get(key_name, shortcut_defaults[key_name])),
                self,
            )
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
    def set_workspace_image(self, image_rgb: np.ndarray | None) -> None:
        self.source_label_path = None
        self.has_image = image_rgb is not None
        if self.has_image:
            self.image = image_rgb
        else:
            self.image = np.full((360, 640, 3), 35, dtype=np.uint8)
        self.annotations.clear()
        self._saved = False
        self._cursor_image_position = None
        height, width = self.image.shape[:2]
        self.source_points[:] = self.source_defaults * np.float32([width - 1, height - 1])
        self.canvas.set_image(self.image)
        self.canvas.setEnabled(self.has_image)
        self.canvas.setVisible(self.has_image)
        self.empty_image.setVisible(not self.has_image)
        self.zoom_status.setText("100%")
        for button in self._image_actions:
            button.setEnabled(self.has_image)
        self.point_tool_button.setEnabled(self.has_image and self.task == "pose")
        self.magic_tool_button.setEnabled(self.has_image and self.task == "segment")
        self._sync_annotation_list()
        self._sync_edit_tools()
        self._sync_source_controls()
        self.reset_history()

    def _make_perspective_controls(self) -> QGroupBox:
        group = QGroupBox("四点（左上、右上、右下、左下）")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("点"), 0, 0)
        grid.addWidget(QLabel("源 X"), 0, 1)
        grid.addWidget(QLabel("源 Y"), 0, 2)
        grid.addWidget(QLabel("目标 X"), 0, 3)
        grid.addWidget(QLabel("目标 Y"), 0, 4)
        for index in range(4):
            grid.addWidget(QLabel(str(index + 1)), index + 1, 0)
            source_row: list[QDoubleSpinBox] = []
            target_row: list[QDoubleSpinBox] = []
            for column, points, point_index in (
                (1, self.source_points, 0),
                (2, self.source_points, 1),
                (3, self.target_points, 0),
                (4, self.target_points, 1),
            ):
                spin = QDoubleSpinBox()
                spin.setDecimals(1)
                spin.setRange(-100000, 100000)
                spin.setSingleStep(1)
                spin.setValue(float(points[index, point_index]))
                spin.valueChanged.connect(self._perspective_control_changed)
                grid.addWidget(spin, index + 1, column)
                (source_row if column < 3 else target_row).append(spin)
            self.source_controls.append(source_row)
            self.target_controls.append(target_row)

        self.output_width = QSpinBox()
        self.output_width.setRange(1, 4096)
        self.output_width.setValue(50)
        self.output_height = QSpinBox()
        self.output_height.setRange(1, 4096)
        self.output_height.setValue(100)
        self.output_width.valueChanged.connect(self._resize_target_points)
        self.output_height.valueChanged.connect(self._resize_target_points)
        sizes = QHBoxLayout()
        sizes.addWidget(QLabel("预览宽"))
        sizes.addWidget(self.output_width)
        sizes.addWidget(QLabel("高"))
        sizes.addWidget(self.output_height)
        container = QWidget()
        container.setLayout(sizes)
        grid.addWidget(container, 5, 0, 1, 5)
        grid.addWidget(
            QLabel("源点默认取当前目标框边界，可拖动调整；越界点显示为灰色。"),
            6,
            0,
            1,
            5,
        )
        return group

    def _select_canvas_tool(self, tool: str) -> None:
        buttons = {
            "box": self.box_tool_button,
            "point": self.point_tool_button,
            "magic": self.magic_tool_button,
            "brush": self.brush_tool_button,
            "eraser": self.eraser_tool_button,
        }
        for name, button in buttons.items():
            button.setChecked(name == tool)
        self.canvas.set_tool(tool)

    def _sync_edit_tools(self) -> None:
        for widget in self.pose_widgets:
            widget.setVisible(self.task == "pose")
        self.segment_tools.setVisible(self.task == "segment")
        self.segment_tools.setEnabled(self.has_image)
        index = self.canvas.selected_box
        selected = self.has_image and self.task == "detect" and 0 <= index < len(self.annotations)
        angle = self.annotations[index].angle if selected else 0.0
        self.angle_edit.setEnabled(selected)
        self.angle_slider.setEnabled(selected)
        self.angle_edit.blockSignals(True)
        self.angle_edit.setValue(angle)
        self.angle_edit.blockSignals(False)
        self._sync_angle_slider(angle)

    def _sync_angle_slider(self, angle: float) -> None:
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(round(angle))
        self.angle_slider.blockSignals(False)

    def _set_angle(self, angle: float) -> None:
        if self.angle_edit.isEnabled():
            self.annotations[self.canvas.selected_box].angle = angle
            self._sync_angle_slider(angle)
            self.canvas.changed.emit()
            self.canvas.update()

    def _segment_shortcut(self, tool: str) -> None:
        if self.has_image and self.task == "segment":
            self._select_canvas_tool(tool)

    def _add_class(self) -> None:
        name, accepted = QInputDialog.getText(self, "新增类别", "类别名称：")
        name = name.strip()
        if not accepted or not name:
            return
        if name in self.classes:
            QMessageBox.warning(self, "类别重复", "该类别名称已存在。")
            return
        self.classes.append(name)
        self.class_combo.blockSignals(True)
        self.class_combo.addItem(name)
        self.class_combo.setCurrentIndex(len(self.classes) - 1)
        self.class_combo.blockSignals(False)
        self.canvas.class_count = len(self.classes)
        self.canvas.set_class(len(self.classes) - 1)
        self.classes_changed.emit(self.classes.copy())
        self.canvas.changed.emit()

    def _delete_class(self) -> None:
        if self.can_delete_class is not None and not self.can_delete_class():
            QMessageBox.warning(self, "删除类别", "数据集正在使用类别编号。请先关闭数据集再删除类别。")
            return
        if len(self.classes) <= 1:
            QMessageBox.warning(self, "无法删除", "至少需要保留一个类别。")
            return
        class_index = self.class_combo.currentIndex()
        if class_index < 0:
            return
        used_count = sum(
            annotation.class_id == class_index for annotation in self.annotations
        )
        if used_count:
            answer = QMessageBox.question(
                self,
                "删除类别",
                f"有 {used_count} 个标注框使用“{self.classes[class_index]}”。"
                "删除后将这些框改为相邻类别，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        del self.classes[class_index]
        for annotation in self.annotations:
            if annotation.class_id == class_index:
                annotation.class_id = min(class_index, len(self.classes) - 1)
            elif annotation.class_id > class_index:
                annotation.class_id -= 1
        self.class_combo.blockSignals(True)
        self.class_combo.removeItem(class_index)
        next_class = min(class_index, len(self.classes) - 1)
        self.class_combo.setCurrentIndex(next_class)
        self.class_combo.blockSignals(False)
        self.canvas.class_count = len(self.classes)
        self.canvas.set_class(next_class)
        if 0 <= self.canvas.selected_box < len(self.annotations):
            self.canvas.class_selected.emit(
                self.annotations[self.canvas.selected_box].class_id
            )
        self.classes_changed.emit(self.classes.copy())
        self.canvas.changed.emit()
        self.edit_history.undo.clear()
        self.edit_history.redo.clear()
        self._update_save_status()
        self.canvas.update()

    def _annotation_list_selection_changed(self, row: int) -> None:
        if not 0 <= row < len(self.annotations):
            return
        self.canvas.selection_locked = True
        self.canvas.selected_box = row
        self.canvas.selected_keypoint = None
        self.canvas.selection_changed.emit(row)
        self.canvas.class_selected.emit(self.annotations[row].class_id)
        self.canvas.changed.emit()
        self.canvas.update()

    def _delete_list_annotation(self) -> None:
        row = self.annotation_list.currentRow()
        if 0 <= row < len(self.annotations) and self.annotation_list.currentItem().isSelected():
            self.canvas.selected_box = row
            self.canvas.delete_selected()

    def _sync_annotation_list(self, *_args) -> None:
        selected = self.canvas.selected_box
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for index, annotation in enumerate(self.annotations):
            class_name = (
                self.classes[annotation.class_id]
                if 0 <= annotation.class_id < len(self.classes)
                else f"类别 {annotation.class_id}"
            )
            left, top, right, bottom = annotation.box
            item = QListWidgetItem(
                f"{index + 1}. {class_name}  "
                f"[{left:.0f}, {top:.0f}, {right:.0f}, {bottom:.0f}]"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.annotation_list.addItem(item)
        if 0 <= selected < self.annotation_list.count():
            self.annotation_list.setCurrentRow(selected)
        self.annotation_list.blockSignals(False)

    def _task_changed(self, index: int) -> None:
        task = self.task_combo.itemData(index)
        if task == self.task:
            return
        if self.annotations:
            answer = QMessageBox.question(
                self,
                "切换标注模式",
                "切换标注模式会清除当前图像的全部标注，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.task_combo.blockSignals(True)
                self.task_combo.setCurrentIndex(
                    self.task_combo.findData(self.task)
                )
                self.task_combo.blockSignals(False)
                return
            self.annotations.clear()
        self.canvas._draft_box = None
        self.canvas._drag_kind = ""
        self.canvas.selected_box = -1
        self.canvas.selected_keypoint = None
        self.canvas.selection_changed.emit(-1)
        self.task = task
        self.canvas.set_task(self.task)
        self.task_changed.emit(self.task)
        self.point_tool_button.setEnabled(self.has_image and self.task == "pose")
        self.magic_tool_button.setEnabled(self.has_image and self.task == "segment")
        if self.task == "segment":
            self._select_canvas_tool("magic")
        elif self.task != "pose" and self.canvas.tool != "box":
            self._select_canvas_tool("box")
        elif self.task == "pose" and 0 <= self.canvas.selected_box < len(self.annotations):
            self._selection_changed(self.canvas.selected_box)
        self._sync_edit_tools()
        self.canvas.changed.emit()

    def _toggle_task(self) -> None:
        self.task_combo.setCurrentIndex((self.task_combo.currentIndex() + 1) % 3)

    def _run_prelabel(self) -> None:
        if not self.has_image:
            return
        model_path = (
            self.model_path.get(self.task)
            if isinstance(self.model_path, dict)
            else self.model_path
        )
        if model_path is None:
            QMessageBox.warning(self, "模型预标注", "请先在设置中选择模型权重。")
            return
        try:
            predictions = predict_annotations(
                self.image, model_path, self.classes, self.task
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as error:
            QMessageBox.critical(self, "模型预标注失败", str(error))
            return
        if not predictions:
            QMessageBox.information(self, "模型预标注", "模型没有检测到目标。")
            return
        self.apply_predictions(predictions)

    def apply_predictions(self, predictions: list[dict]) -> None:
        if not predictions:
            return
        previous = self.canvas.selected_box
        for prediction in predictions:
            polygon = [
                (float(point[0]), float(point[1]))
                for point in prediction.get("polygon", [])
            ]
            self.annotations.append(
                Annotation(
                    [float(value) for value in prediction["box"]],
                    int(prediction["class_id"]),
                    list(prediction.get("keypoints", [])),
                    polygon,
                )
            )
        self.canvas.selected_box = len(self.annotations) - 1
        self.canvas.selected_keypoint = None
        if previous != self.canvas.selected_box:
            self.canvas.selection_changed.emit(self.canvas.selected_box)
        self.canvas.class_selected.emit(self.annotations[-1].class_id)
        self.canvas.changed.emit()
        self.canvas.update()

    def _class_changed(self, index: int) -> None:
        if index >= 0:
            self.canvas.set_class(index)
        if 0 <= self.canvas.selected_box < len(self.annotations) and index >= 0:
            self.annotations[self.canvas.selected_box].class_id = index
            self.canvas.changed.emit()
            self.canvas.update()

    def _select_class(self, class_id: int) -> None:
        if 0 <= class_id < self.class_combo.count():
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(class_id)
            self.class_combo.blockSignals(False)
            self.canvas.set_class(class_id)

    def _selection_changed(self, index: int) -> None:
        self._sync_edit_tools()
        if 0 <= index < len(self.annotations):
            left, top, right, bottom = self.annotations[index].box
            self.source_points[:] = np.float32([left, top]) + self.source_defaults * np.float32(
                [right - left, bottom - top]
            )
            self._sync_source_controls()
        else:
            self._refresh_preview()

    def _selected_box_geometry_changed(
        self, index: int, old_box: list[float], new_box: list[float]
    ) -> None:
        if index != self.canvas.selected_box:
            return
        old_width, old_height = old_box[2] - old_box[0], old_box[3] - old_box[1]
        new_width, new_height = new_box[2] - new_box[0], new_box[3] - new_box[1]
        if old_width <= 0 or old_height <= 0 or new_width <= 0 or new_height <= 0:
            return
        normalized = (self.source_points - np.float32(old_box[:2])) / np.float32(
            [old_width, old_height]
        )
        self.source_points[:] = np.float32(new_box[:2]) + normalized * np.float32(
            [new_width, new_height]
        )
        self._sync_source_controls()

    def _sync_source_controls(self) -> None:
        for index in range(4):
            for column in range(2):
                control = self.source_controls[index][column]
                control.blockSignals(True)
                control.setValue(float(self.source_points[index, column]))
                control.blockSignals(False)
        self._refresh_preview()

    def _perspective_control_changed(self, _value: float) -> None:
        for index in range(4):
            self.source_points[index] = [
                self.source_controls[index][0].value(),
                self.source_controls[index][1].value(),
            ]
            self.target_points[index] = [
                self.target_controls[index][0].value(),
                self.target_controls[index][1].value(),
            ]
        self.canvas.update()
        self._refresh_preview()

    def _target_preview_point_dragged(self, index: int, x: float, y: float) -> None:
        self.target_controls[index][0].setValue(x)
        self.target_controls[index][1].setValue(y)

    def _resize_target_points(self, _value: int) -> None:
        width, height = self.output_width.value(), self.output_height.value()
        points = self.target_defaults * np.float32([width, height])
        for index, point in enumerate(points):
            for column in range(2):
                self.target_controls[index][column].blockSignals(True)
                self.target_controls[index][column].setValue(float(point[column]))
                self.target_controls[index][column].blockSignals(False)
                self.target_points[index, column] = point[column]
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if not self.has_image:
            self._preview_pixmap = QPixmap()
            self._preview_image = QImage()
            self._magnifier_image = QImage()
            self.preview.clear()
            self.preview_status.clear()
            self.preview_magnifier.clear()
            self.preview_magnifier.hide()
            return
        if self.task != "pose":
            height, width = self.image.shape[:2]
            self._magnifier_image = QImage(
                self.image.data,
                width,
                height,
                width * 3,
                QImage.Format.Format_RGB888,
            ).copy()
            self._preview_image = QImage()
            self.preview.clear()
            self.preview.setText("透视预览仅用于 pose 模式")
            self.preview_status.setText("detect 与 segment 模式的放大镜显示原图区域。")
            self._preview_pixmap = QPixmap()
            self._update_preview_magnifier()
            return
        if not 0 <= self.canvas.selected_box < len(self.annotations):
            self._preview_image = QImage()
            self._magnifier_image = QImage()
            self.preview_magnifier.clear()
            self.preview.clear()
            self.preview.setText("选择 pose 目标框以查看透视预览")
            self.preview_status.setText(
                "透视源点以当前选中目标框为基准；切换目标后会重新定位。"
            )
            self._preview_pixmap = QPixmap()
            return
        try:
            matrix = build_homography(self.source_points, self.target_points)
        except ValueError as error:
            self._preview_image = QImage()
            self._magnifier_image = QImage()
            self.preview_magnifier.clear()
            self.preview.clear()
            self.preview.setText("透视参数无效")
            self.preview_status.setText(str(error))
            self._preview_pixmap = QPixmap()
            return

        output_width, output_height = self.output_width.value(), self.output_height.value()
        warped = cv2.warpPerspective(
            self.image,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(35, 35, 35),
        )
        magnifier_image = QImage(
            warped.data,
            warped.shape[1],
            warped.shape[0],
            warped.shape[1] * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        height, width = warped.shape[:2]
        out_of_bounds = 0
        annotation = self.annotations[self.canvas.selected_box]
        try:
            polygon = map_box(tuple(annotation.box), matrix)
            left, top = np.min(polygon, axis=0)
            right, bottom = np.max(polygon, axis=0)
            cv2.rectangle(
                warped,
                (int(round(float(left))), int(round(float(top)))),
                (int(round(float(right))), int(round(float(bottom)))),
                tuple(self.box_color.getRgb()[:3]),
                1,
                cv2.LINE_AA,
            )
        except cv2.error:
            pass
        mapped_points: list[tuple[float, float]] = []
        for point in annotation.keypoints:
            if point is None:
                continue
            mapped = map_points(np.float32([point]), matrix)[0]
            is_inside = 0 <= mapped[0] < width and 0 <= mapped[1] < height
            out_of_bounds += not is_inside
            mapped_points.append((float(mapped[0]), float(mapped[1])))
        for first, second in _closed_edges(mapped_points):
            cv2.line(
                warped,
                (int(round(first[0])), int(round(first[1]))),
                (int(round(second[0])), int(round(second[1]))),
                (80, 190, 255),
                1,
                cv2.LINE_AA,
            )
        for point in mapped_points:
            is_inside = 0 <= point[0] < width and 0 <= point[1] < height
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            point_color = (80, 190, 255) if is_inside else (160, 160, 160)
            cv2.circle(warped, (x, y), 2, point_color, -1, cv2.LINE_AA)
        for index, point in enumerate(self.target_points):
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            cv2.circle(warped, (x, y), 4, (255, 210, 80), -1, cv2.LINE_AA)
            cv2.putText(
                warped,
                f"T{index + 1}",
                (min(x + 5, width - 18), max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 210, 80),
                1,
                cv2.LINE_AA,
            )

        qimage = QImage(
            warped.data, width, height, width * 3, QImage.Format.Format_RGB888
        ).copy()
        self._preview_image = qimage
        self._magnifier_image = magnifier_image
        self._preview_pixmap = QPixmap.fromImage(qimage)
        self.preview.set_target_geometry(self.target_points, width, height)
        self._show_preview_pixmap()
        self._update_preview_magnifier()
        self.preview_status.setText(
            f"H 已实时计算。映射出界关键点：{out_of_bounds}"
            if out_of_bounds
            else "H 已实时计算。标注坐标仍以原图为准。"
        )

    def _show_preview_pixmap(self) -> None:
        if not self._preview_pixmap.isNull():
            self.preview.setPixmap(
                self._preview_pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._update_preview_magnifier()

    def _pointer_position_changed(self, position) -> None:
        if position is None:
            return
        self._cursor_image_position = position
        self._update_preview_magnifier()

    def _update_preview_magnifier(self) -> None:
        if (
            self._magnifier_image.isNull() or self._cursor_image_position is None
        ):
            return
        if self.task == "pose":
            if not 0 <= self.canvas.selected_box < len(self.annotations):
                return
            try:
                matrix = build_homography(self.source_points, self.target_points)
            except ValueError:
                return
            mapped = map_points(np.float32([self._cursor_image_position]), matrix)[0]
            reference_size = (self.preview.width(), self.preview.height())
        else:
            mapped = np.asarray(self._cursor_image_position, dtype=np.float32)
            reference_size = (self.canvas.width(), self.canvas.height())
        width, height = self._magnifier_image.width(), self._magnifier_image.height()
        if not (0 <= mapped[0] < width and 0 <= mapped[1] < height):
            return
        self.preview_magnifier.set_content(
            self._magnifier_image,
            (float(mapped[0]), float(mapped[1])),
            self.magnifier_zoom,
            reference_size,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show_preview_pixmap()

    def done(self, result: int) -> None:
        self.preview_magnifier.hide()
        super().done(result)

    def reject(self) -> None:
        if self.confirm_close is not None and not self.confirm_close():
            return
        self.preview_magnifier.hide()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.preview_magnifier.hide()
        super().closeEvent(event)

    def _request_save(self) -> None:
        if not self.has_image:
            return
        if not self.annotations and self.source_label_path is None:
            QMessageBox.warning(self, "没有标注", "请至少框选一个目标后再保存。")
            return
        for annotation in self.annotations:
            if annotation.box[2] <= annotation.box[0] or annotation.box[3] <= annotation.box[1]:
                QMessageBox.warning(self, "标注无效", "所有目标框必须具有正的宽度和高度。")
                return
            if not 0 <= annotation.class_id < len(self.classes):
                QMessageBox.warning(self, "类别无效", "请为每个目标选择有效类别。")
                return
            if self.task == "segment" and len(annotation.polygon) < 3:
                QMessageBox.warning(self, "分割标注无效", "每个分割实例至少需要 3 个多边形点。")
                return
        try:
            matrix = (
                build_homography(self.source_points, self.target_points)
                if self.task == "pose" and self.source_label_path is None else np.eye(3)
            )
        except ValueError as error:
            QMessageBox.warning(self, "透视参数无效", str(error))
            return
        serialized = [
            {
                "box": annotation.box.copy(),
                "angle": annotation.angle,
                "class_id": annotation.class_id,
                "keypoints": [
                    list(point) if point is not None else None
                    for point in annotation.keypoints
                ],
                "polygon": [list(point) for point in annotation.polygon],
                "keypoint_visibility": annotation.keypoint_visibility.copy(),
            }
            for annotation in self.annotations
        ]
        self.save_requested.emit(
            {
                "task": self.task,
                "annotations": serialized,
                "source_points": self.source_points.copy(),
                "target_points": self.target_points.copy(),
                "homography": matrix,
                "perspective_target": (
                    {
                        "annotation_index": self.canvas.selected_box,
                        "box": self.annotations[self.canvas.selected_box].box.copy(),
                    }
                    if self.task == "pose"
                    else None
                ),
            }
        )

    def mark_saved(self) -> None:
        self.finish_edit()
        self.edit_history.saved = self.edit_history.current
        self.edit_history.has_saved_label = True
        self._update_save_status()
        self._saved = True
        if getattr(self, "persistent_workspace", False):
            self.setWindowTitle(self.windowTitle().removesuffix(" · 已保存") + " · 已保存")
        else:
            self.accept()
