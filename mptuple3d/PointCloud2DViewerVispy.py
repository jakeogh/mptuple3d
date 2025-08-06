#!/usr/bin/env python3
# tab-width:4

# pylint: disable=no-name-in-module
# pylint: disable=no-member

from __future__ import annotations

from signal import SIG_DFL
from signal import SIGPIPE
from signal import signal
from time import time

import numpy as np
from asserttool import icp
from PyQt6.QtCore import Qt  # type: ignore
from PyQt6.QtGui import QGuiApplication  # type: ignore
from PyQt6.QtWidgets import QCheckBox  # type: ignore
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from vispy import app  # type: ignore
from vispy import scene
from vispy.app import use_app  # type: ignore
from vispy.scene import visuals  # type: ignore
from vispy.scene.cameras import PanZoomCamera  # type: ignore

from .ColorManager import ColorManager
from .ColorManager import make_colors_from_scalar
from .InputState import InputState
from .utils import KeyboardInputManager
from .utils import center_points_2d
from .utils import get_rect_from_points
from .utils import normalize_points_2d
from .utils import pad_to_3d

use_app("pyqt6")

# keep SIGPIPE default behavior
signal(SIGPIPE, SIG_DFL)


# =========================
# 2D Viewer
# =========================
class PointCloud2DViewerVispy(QMainWindow):
    """
    Pure 2D viewer with Pan/Zoom camera.
    - Keeps X/Y axis scaling via X / Shift+X and Y / Shift+Y keys (like 3D).
    - Ignores Z; optional scalar colors supported.
    """

    def __init__(
        self,
        points_xy: np.ndarray,  # (N,2)
        color_data: np.ndarray | None = None,
        normalize: bool = True,
        disable_antialiasing: bool = False,
        draw_lines: bool = False,
        size: float | None = None,
    ):
        super().__init__()
        if points_xy.shape[0] == 0:
            raise ValueError("No valid 2D points loaded")

        # Use consolidated utility functions
        proc = (
            normalize_points_2d(points_xy) if normalize else center_points_2d(points_xy)
        )

        self.state = InputState()
        self.last_time = time()
        self.points2d = proc
        self.color_data = color_data
        self.disable_antialiasing = disable_antialiasing
        self.draw_lines = draw_lines
        self.acceleration = 1.1

        # Create managers early, before any method calls that might use them
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)
        self.color_manager = ColorManager(
            backend="vispy", colormap="viridis", default_color="white"
        )

        self.canvas = scene.SceneCanvas(keys="interactive", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.canvas.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.native.setFocus()

        # 2D scene: PanZoom camera with fixed depth
        cam = PanZoomCamera(aspect=None)  # allow independent x/y zoom
        cam.rect = get_rect_from_points(self.points2d)  # Use consolidated function
        self.view.camera = cam
        # Disable default camera mouse interactions so we can remap:
        cam.interactive = False

        # --- Box-zoom (left-drag) and Pan (Shift+left-drag) state ---
        self._mode = None  # 'select' or 'pan' or None
        self._sel_start_data = None
        self._last_canvas_pos = None
        self._rect_vis = None

        # Hook mouse events on the viewbox so we intercept before camera
        vb = self.view.camera.viewbox
        vb.events.mouse_press.connect(self._vb_mouse_press)
        vb.events.mouse_move.connect(self._vb_mouse_move)
        vb.events.mouse_release.connect(self._vb_mouse_release)

        self.scatter = visuals.Markers()
        point_count = len(self.points2d)
        if size is not None:
            self.point_size = size
        elif point_count > 100000:
            self.point_size = 3
        else:
            self.point_size = 5

        self.use_antialiasing = (point_count <= 100000) and (not disable_antialiasing)
        if disable_antialiasing:
            self.use_antialiasing = False

        self.scatter.set_data(
            pos=pad_to_3d(self.points2d),  # Use consolidated function
            face_color=self._make_colors(),
            size=self.point_size,
            edge_color=None,
            edge_width=0,
        )

        self.line = None
        if self.draw_lines:
            self.line = visuals.Line(
                color="gray",
                width=1,
                connect="strip",
                method="gl",
            )
            self.line.set_data(pos=pad_to_3d(self.points2d * self.state.scale[:2]))
            self.view.add(self.line)

        if hasattr(self.scatter, "antialias"):
            self.scatter.antialias = 1.0 if self.use_antialiasing else 0.0

        self.view.add(self.scatter)

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.timer = app.Timer("auto", connect=self.on_timer, start=True)

        central = QWidget()
        main_layout = QVBoxLayout()
        # Fill vertical space with the canvas; remove dead padding
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)
        main_layout.addWidget(self.canvas.native, 1)  # stretch the canvas

        control_row = QHBoxLayout()
        control_row.setContentsMargins(
            8,
            4,
            8,
            4,
        )  # compact control strip
        control_row.setSpacing(6)

        accel_label = QLabel("Accel:")
        control_row.addWidget(accel_label)

        self.accel_spinbox = QDoubleSpinBox()
        self.accel_spinbox.setRange(1.001, 5.0)
        self.accel_spinbox.setSingleStep(0.01)
        self.accel_spinbox.setDecimals(3)
        self.accel_spinbox.setValue(self.acceleration)
        self.accel_spinbox.valueChanged.connect(self.on_acceleration_changed)
        self.accel_spinbox.setMaximumWidth(80)
        control_row.addWidget(self.accel_spinbox)

        size_label = QLabel("Size:")
        control_row.addWidget(size_label)

        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.1, 1000.0)
        self.size_spinbox.setSingleStep(0.1)
        self.size_spinbox.setDecimals(3)
        self.size_spinbox.setValue(self.point_size)
        self.size_spinbox.valueChanged.connect(self.on_point_size_changed)
        self.size_spinbox.setMaximumWidth(70)
        control_row.addWidget(self.size_spinbox)

        self.aa_checkbox = QCheckBox("AA")
        self.aa_checkbox.setChecked(self.use_antialiasing)
        self.aa_checkbox.toggled.connect(self.on_antialiasing_changed)
        self.aa_checkbox.setToolTip("Antialiasing")
        control_row.addWidget(self.aa_checkbox)

        self.lines_checkbox = QCheckBox("Lines")
        self.lines_checkbox.setChecked(self.draw_lines)
        self.lines_checkbox.toggled.connect(self.on_lines_toggled)
        control_row.addWidget(self.lines_checkbox)

        info_text = f"{len(self.points2d):,} pts"
        self.info_label = QLabel(info_text)
        control_row.addWidget(self.info_label)

        main_layout.addLayout(control_row)
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("2D Point Viewer")

        print(f"[INFO] Loaded {point_count:,} points (2D)")
        print(f"[INFO] Using point size: {self.point_size}")
        print(
            f"[INFO] Antialiasing: {'enabled' if self.use_antialiasing else 'disabled'}"
        )

    def _make_colors(self):
        # Use consolidated color function
        return self.color_manager.make_colors(self.color_data)

    def on_lines_toggled(self, checked: bool):
        self.draw_lines = checked
        if checked:
            self.line = visuals.Line(
                color="gray",
                width=1,
                connect="strip",
                method="gl",
            )
            self.line.set_data(pos=pad_to_3d(self.points2d * self.state.scale[:2]))
            self.view.add(self.line)
        else:
            if self.line is not None:
                self.view.remove(self.line)
                self.line = None

    def on_acceleration_changed(self, value: float):
        self.acceleration = value
        self.keyboard_manager.set_acceleration(value)

    def on_point_size_changed(self, value: float):
        self.point_size = value
        scaled_xy = self.points2d * self.state.scale[:2]
        self.scatter.set_data(
            pad_to_3d(scaled_xy), face_color=self._make_colors(), size=self.point_size
        )

    def on_antialiasing_changed(self, checked: bool):
        self.use_antialiasing = checked
        if hasattr(self.scatter, "antialias"):
            self.scatter.antialias = 1.0 if checked else 0.0
            self.canvas.update()

    def on_key_press(self, event):
        key_name = event.key.name if event.key else "None"
        has_shift = "Shift" in event.modifiers
        self.state.add_key(key_name, has_shift)
        if key_name in {"Q", "Escape"} and not has_shift:
            print("[INFO] 'q' pressed, closing viewer.")
            self.close()

    def on_key_release(self, event):
        key_name = event.key.name if event.key else "None"
        self.state.remove_key(key_name)

    def on_timer(self, _):
        frame_start = time()
        now = time()
        dt = now - self.last_time
        self.last_time = now

        self._update_scaling(dt)
        scaled_xy = self.points2d * self.state.scale[:2]
        self.scatter.set_data(
            pad_to_3d(scaled_xy), face_color=self._make_colors(), size=self.point_size
        )
        if self.draw_lines and self.line is not None:
            self.line.set_data(pos=pad_to_3d(scaled_xy))

    def _update_scaling(self, dt: float):
        # Use consolidated keyboard manager with lowercase behavior - matches exact original behavior
        self.keyboard_manager.update_scaling_2d_lowercase(dt)

    # --- Box-zoom & Shift-pan (2D only) -------------------------------------------------
    def _shift_down(self) -> bool:
        """Robust Shift detection via Qt (VisPy SceneMouseEvent may lack .modifiers)."""
        try:
            mods = QGuiApplication.keyboardModifiers()
            return bool(mods & Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            return False

    def _canvas_to_data(self, pos):
        p = self.view.scene.transform.imap(pos)
        return float(p[0]), float(p[1])

    def _ensure_rect_vis(self, center):
        # vispy.Rectangle can't start at 0x0 (division by zero when computing segments)
        eps = 1e-6
        if self._rect_vis is None:
            self._rect_vis = visuals.Rectangle(
                center=center,
                width=eps,
                height=eps,
                radius=0.0,  # no rounded corners
                color=None,
                border_color="white",
            )
            self.view.add(self._rect_vis)
        else:
            self._rect_vis.center = center
            self._rect_vis.width = eps
            self._rect_vis.height = eps
            self._rect_vis.visible = True

    def _vb_mouse_press(self, event):
        if event.button == 1 and self._shift_down():
            # Begin panning
            self._mode = "pan"
            self._last_canvas_pos = event.pos
            event.handled = True
            return
        if event.button == 1:
            # Begin box selection
            self._mode = "select"
            x0, y0 = self._canvas_to_data(event.pos)
            self._sel_start_data = (x0, y0)
            self._ensure_rect_vis((x0, y0))
            event.handled = True

    def _vb_mouse_move(self, event):
        if self._mode == "pan" and self._last_canvas_pos is not None:
            # Convert pixel delta to data rect shift
            rect = self.view.camera.rect
            # Access rect properties directly instead of converting to list
            rect_left = rect.left
            rect_bottom = rect.bottom
            rect_width = rect.width
            rect_height = rect.height

            vp_w, vp_h = self.canvas.size
            dx = event.pos[0] - self._last_canvas_pos[0]
            dy = event.pos[1] - self._last_canvas_pos[1]
            sx = rect_width / max(vp_w, 1)
            sy = rect_height / max(vp_h, 1)

            # Calculate new rect values
            new_left = rect_left - dx * sx
            new_bottom = rect_bottom + dy * sy  # y pixels increase downward

            # Set the new rect using a tuple
            self.view.camera.rect = (new_left, new_bottom, rect_width, rect_height)
            self._last_canvas_pos = event.pos
            event.handled = True
        elif self._mode == "select" and self._sel_start_data is not None:
            x0, y0 = self._sel_start_data
            x1, y1 = self._canvas_to_data(event.pos)
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            w, h = abs(x1 - x0), abs(y1 - y0)
            self._ensure_rect_vis((cx, cy))

            eps = 1e-6
            self._rect_vis.width = max(w, eps)
            self._rect_vis.height = max(h, eps)

            event.handled = True

    def _vb_mouse_release(self, event):
        if event.button != 1:
            return
        if self._mode == "select" and self._sel_start_data is not None:
            x0, y0 = self._sel_start_data
            x1, y1 = self._canvas_to_data(event.pos)

            if self._rect_vis is not None:
                # ViewBox has no .remove(); detach visual by clearing its parent
                self._rect_vis.parent = None
                self._rect_vis = None

            w, h = abs(x1 - x0), abs(y1 - y0)
            rect = self.view.camera.rect
            # Rect is a vispy.geometry.Rect (not subscriptable here)
            min_w = float(rect.width) * 1e-3
            min_h = float(rect.height) * 1e-3
            icp(min_w, min_h)

            if w >= min_w and h >= min_h:
                xmin, ymin = min(x0, x1), min(y0, y1)
                icp(xmin, ymin)
                pad_x, pad_y = w * 0.02, h * 0.02
                self.view.camera.rect = (
                    xmin - pad_x,
                    ymin - pad_y,
                    w + 2 * pad_x,
                    h + 2 * pad_y,
                )
            # Ensure the new view is applied immediately
            self.canvas.update()
            event.handled = True

        elif self._mode == "pan":
            event.handled = True
        self._mode = None
        self._sel_start_data = None
        self._last_canvas_pos = None

    def show_gui(self):
        self.show()
        app.run()
