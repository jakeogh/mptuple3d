#!/usr/bin/env python3
# tab-width:4

# py lint: disable=useless-suppression             # [I0021]
# pylint: disable=too-many-instance-attributes    # [R0902]
# pylint: disable=invalid-name                    # [C0103] single letter var names, name too descriptive(!)
# pylint: disable=no-member                       # [E1101] no member for base

# pylint: disable=no-name-in-module

from __future__ import annotations

from time import time

import numpy as np
from PyQt6.QtCore import Qt  # type: ignore
from PyQt6.QtWidgets import QCheckBox  # type: ignore
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from vispy import app  # type: ignore
from vispy import scene
from vispy.scene import visuals  # type: ignore

from .ColorManager import ColorManager
from .InputState import InputState
from .mptuple3d import UnlimitedTurntableCamera
from .utils import KeyboardInputManager
from .utils import center_points
from .ColorManager import make_colors_from_scalar
from .utils import normalize_points


# =========================
# 3D Viewer
# =========================
class PointCloud3DViewerVispy(QMainWindow):
    def __init__(
        self,
        points: np.ndarray,
        color_data: np.ndarray | None = None,
        normalize: bool = True,
        view_mode: str | None = None,
        disable_antialiasing: bool = False,
        draw_lines: bool = False,
        size: float | None = None,
    ):
        super().__init__()
        if points.shape[0] == 0:
            raise ValueError("No valid points loaded")

        # Use consolidated utility functions
        processed_points = (
            normalize_points(points) if normalize else center_points(points)
        )

        self.state = InputState()
        self.last_time = time()
        self.points = processed_points
        self.color_data = color_data
        self.disable_antialiasing = disable_antialiasing
        self.draw_lines = draw_lines

        # Add acceleration parameter that can be adjusted
        self.acceleration = 1.1

        # Create managers early, before any method calls that might use them
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)
        self.color_manager = ColorManager(
            backend="vispy", colormap="viridis", default_color="white"
        )

        # canvas and scene
        self.canvas = scene.SceneCanvas(keys="interactive", show=False)
        self.view = self.canvas.central_widget.add_view()
        # focus fix
        self.canvas.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.native.setFocus()

        # Estimate distance to fit all points
        bbox_min = processed_points.min(axis=0)
        bbox_max = processed_points.max(axis=0)
        bbox_size = np.linalg.norm(bbox_max - bbox_min)
        initial_distance = bbox_size * 1.2 if bbox_size > 0 else 2.5

        self.camera = UnlimitedTurntableCamera(fov=45, distance=initial_distance)
        self.view.camera = self.camera
        self.view.camera.interactive = True

        # Use optimized settings for large point clouds
        self.scatter = visuals.Markers()

        # Determine optimal settings based on point count and CLI args
        point_count = len(self.points)
        if size is not None:
            self.point_size = size
        elif point_count > 100000:
            self.point_size = 5
        else:
            self.point_size = 5

        if point_count > 100000:
            self.use_antialiasing = False
        else:
            self.use_antialiasing = not disable_antialiasing

        # Override antialiasing if explicitly disabled
        if disable_antialiasing:
            self.use_antialiasing = False

        self.scatter.set_data(
            self.points,
            face_color=self._make_colors(),
            size=self.point_size,
            edge_color=None,  # Disable edges for performance
            edge_width=0,  # No edge width for performance
        )

        self.line = None
        if self.draw_lines:
            self.line = visuals.Line(
                color="gray",
                width=1,
                connect="strip",
                method="gl",
            )
            self.line.set_data(pos=self.points * self.state.scale)
            self.view.add(self.line)

        # Set antialiasing
        if hasattr(self.scatter, "antialias"):
            self.scatter.antialias = 1.0 if self.use_antialiasing else 0.0

        self.view.add(self.scatter)

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.timer = app.Timer("auto", connect=self.on_timer, start=True)

        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas.native)

        # Single row with all controls
        control_row = QHBoxLayout()

        # Acceleration control
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

        # Point size control
        size_label = QLabel("Size:")
        control_row.addWidget(size_label)

        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.1, 1000.0)  # 0 to inf (practically)
        self.size_spinbox.setSingleStep(0.1)
        self.size_spinbox.setDecimals(3)
        self.size_spinbox.setValue(self.point_size)
        self.size_spinbox.valueChanged.connect(self.on_point_size_changed)
        self.size_spinbox.setMaximumWidth(70)
        control_row.addWidget(self.size_spinbox)

        # Antialiasing checkbox
        self.aa_checkbox = QCheckBox("AA")
        self.aa_checkbox.setChecked(self.use_antialiasing)
        self.aa_checkbox.toggled.connect(self.on_antialiasing_changed)
        self.aa_checkbox.setToolTip("Antialiasing")
        control_row.addWidget(self.aa_checkbox)

        self.lines_checkbox = QCheckBox("Lines")
        self.lines_checkbox.setChecked(self.draw_lines)
        self.lines_checkbox.toggled.connect(self.on_lines_toggled)
        control_row.addWidget(self.lines_checkbox)

        # Point count
        info_text = f"{len(self.points):,} pts"
        self.info_label = QLabel(info_text)
        control_row.addWidget(self.info_label)

        # Add stretch to push view buttons to the right
        control_row.addStretch()

        # View buttons on the right
        for mode in ("XY", "XZ", "YZ"):
            button = QPushButton(mode)
            button.clicked.connect(self.make_button_handler(mode.lower()))
            button.setMaximumWidth(40)
            control_row.addWidget(button)

        main_layout.addLayout(control_row)
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("3D Point Viewer")

        # Print info
        print(f"[INFO] Loaded {point_count:,} points")
        print(f"[INFO] Using point size: {self.point_size}")
        print(
            f"[INFO] Antialiasing: {'enabled' if self.use_antialiasing else 'disabled'}"
        )

        # optional view mode on launch
        if view_mode:
            self.set_view(view_mode)

    def on_lines_toggled(self, checked: bool):
        self.draw_lines = checked
        if checked:
            self.line = visuals.Line(
                color="gray",
                width=1,
                connect="strip",
                method="gl",
            )
            self.line.set_data(pos=self.points * self.state.scale)
            self.view.add(self.line)
        else:
            if self.line is not None:
                self.view.remove(self.line)
                self.line = None

    def make_button_handler(self, mode: str):
        def handler():
            self.set_view(mode)
            self.canvas.native.setFocus()

        return handler

    def on_acceleration_changed(self, value: float):
        """Called when acceleration spinbox value changes"""
        self.acceleration = value
        self.keyboard_manager.set_acceleration(value)

    def on_point_size_changed(self, value: float):
        """Called when point size spinbox value changes"""
        self.point_size = value
        # Trigger immediate redraw with new size
        scaled_points = self.points * self.state.scale
        self.scatter.set_data(
            scaled_points, face_color=self._make_colors(), size=self.point_size
        )

    def on_antialiasing_changed(self, checked: bool):
        """Called when antialiasing checkbox changes"""
        self.use_antialiasing = checked
        if hasattr(self.scatter, "antialias"):
            self.scatter.antialias = 1.0 if checked else 0.0
            # Force redraw
            self.canvas.update()

    def _make_colors(self):
        # Use consolidated color function
        return self.color_manager.make_colors(self.color_data)

    def set_view(self, mode: str):
        if mode == "xy":
            self.camera.azimuth = 0
            self.camera.elevation = 90
        elif mode == "xz":
            self.camera.azimuth = 0
            self.camera.elevation = 0
        elif mode == "yz":
            self.camera.azimuth = 90
            self.camera.elevation = 0
        print(f"[INFO] View set to {mode.upper()}")

    def on_key_press(self, event):
        key_name = event.key.name if event.key else "None"
        has_shift = "Shift" in event.modifiers
        self.state.add_key(key_name, has_shift)

        # Exit on plain 'q'
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
        self._update_rotation(dt)
        scaled_points = self.points * self.state.scale
        self.scatter.set_data(
            scaled_points, face_color=self._make_colors(), size=self.point_size
        )
        if self.draw_lines and self.line is not None:
            self.line.set_data(pos=scaled_points)

    def _update_scaling(self, dt: float):
        # Use consolidated keyboard manager - matches exact behavior for 3D (X, Y, Z)
        self.keyboard_manager.update_scaling(dt, dimensions=3)

    def _update_rotation(self, dt: float):
        rot_speed = 90.0 * dt
        rot_changed = False
        for key in self.state.shift_keys:
            if key == "Left":
                self.state.rotation[1] -= rot_speed
                rot_changed = True
            elif key == "Right":
                self.state.rotation[1] += rot_speed
                rot_changed = True
            elif key == "Up":
                self.state.rotation[0] -= rot_speed
                rot_changed = True
            elif key == "Down":
                self.state.rotation[0] += rot_speed
                rot_changed = True
            elif key == "Q":
                self.state.rotation[2] -= rot_speed
                rot_changed = True
            elif key == "E":
                self.state.rotation[2] += rot_speed
                rot_changed = True

        if rot_changed:
            pitch, yaw, roll = self.state.rotation
            self.camera.elevation = pitch
            self.camera.azimuth = yaw
            self.camera.roll = roll
        else:
            self.state.rotation[0] = self.camera.elevation
            self.state.rotation[1] = self.camera.azimuth
            self.state.rotation[2] = self.camera.roll

    def show_gui(self):
        self.show()
        app.run()
