#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from time import time
from typing import Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .InputState import InputState
from .ColorManager import ColorManager
from .utils import KeyboardInputManager
from .utils import center_points
from .ColorManager import make_colors_from_scalar
from .utils import normalize_points


class PointCloud3DViewerMatplotlib(QMainWindow):
    """
    Matplotlib-based 3D point cloud viewer with interactive controls.

    Features:
    - 3D scatter plot with rotatable view
    - Mouse interaction for rotation and zoom
    - Keyboard scaling controls (X/Shift+X, Y/Shift+Y, Z/Shift+Z like VisPy version)
    - Performance optimizations with downsampling for large datasets
    - Optional line connections between points
    - View preset buttons (XY, XZ, YZ)
    - ESC or Q to exit immediately
    """

    def __init__(
        self,
        points_xyz: np.ndarray,
        color_data: np.ndarray | None = None,
        normalize: bool = True,
        disable_antialiasing: bool = False,
        draw_lines: bool = False,
        size: float | None = None,
        view_mode: str | None = None,
    ):
        super().__init__()

        # Validate input
        if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
            raise ValueError(
                "Input array must be 2D with at least 3 columns (X, Y, Z)."
            )
        if points_xyz.shape[0] == 0:
            raise ValueError("No valid points loaded")

        # Process points using consolidated utility functions
        self.original_points = points_xyz[:, :3].astype(np.float32)
        self.color_data = color_data

        if normalize:
            self.points3d = normalize_points(self.original_points)
        else:
            self.points3d = center_points(self.original_points)

        # Store normalize flag for proper behavior
        self.normalize_mode = normalize

        # State management
        self.state = InputState()
        self.last_time = time()
        self.disable_antialiasing = disable_antialiasing
        self.draw_lines = draw_lines
        self.acceleration = 1.1

        # Create managers early, before any method calls that might use them
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)
        self.color_manager = ColorManager(
            backend="matplotlib", colormap="viridis", default_color="white"
        )

        # Point size
        point_count = len(self.points3d)
        if size is not None:
            self.point_size = size
        elif point_count > 100000:
            self.point_size = 3
        else:
            self.point_size = 5

        # Setup matplotlib figure with 3D projection
        self.fig = Figure(facecolor="black")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Tight layout to maximize plot area
        self.fig.subplots_adjust(left=0.04, right=0.97, top=0.97, bottom=0.04)

        # Style the 3D plot
        self.ax.set_facecolor("black")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Make pane edges white for visibility
        self.ax.xaxis.pane.set_edgecolor("white")
        self.ax.yaxis.pane.set_edgecolor("white")
        self.ax.zaxis.pane.set_edgecolor("white")
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        # Set equal aspect ratio
        self._set_equal_aspect_3d()

        # Initial downsample for performance
        self.max_display_points = 5000
        self.current_downsample = max(1, len(self.points3d) // self.max_display_points)

        # Plot initial data
        self._update_plot()

        # Connect key events to the main window AND the canvas
        self.canvas.mpl_connect("key_press_event", self.on_matplotlib_key_press)

        # Setup UI
        self._setup_ui()

        # Keyboard and timer setup
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()

        # Timer for continuous updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)  # ~60 FPS

        # Set initial view mode if specified
        if view_mode:
            self.set_view(view_mode)

        print(f"[INFO] Loaded {point_count:,} points (3D Matplotlib)")
        print(f"[INFO] Using point size: {self.point_size}")
        print(f"[INFO] Normalize mode: {'enabled' if normalize else 'disabled'}")
        print(
            f"[INFO] Antialiasing: {'enabled' if not disable_antialiasing else 'disabled'}"
        )

    def _setup_ui(self):
        """Setup the UI layout with controls."""
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Add canvas with stretch
        main_layout.addWidget(self.canvas, 1)

        # Control row
        control_row = QHBoxLayout()
        control_row.setContentsMargins(8, 4, 8, 4)
        control_row.setSpacing(6)

        # Acceleration control
        accel_label = QLabel("Accel:")
        accel_label.setStyleSheet("color: white;")
        control_row.addWidget(accel_label)

        self.accel_spinbox = QDoubleSpinBox()
        self.accel_spinbox.setRange(1.001, 5.0)
        self.accel_spinbox.setSingleStep(0.01)
        self.accel_spinbox.setDecimals(3)
        self.accel_spinbox.setValue(self.acceleration)
        self.accel_spinbox.valueChanged.connect(self.on_acceleration_changed)
        self.accel_spinbox.setMaximumWidth(80)
        control_row.addWidget(self.accel_spinbox)

        # Size control
        size_label = QLabel("Size:")
        size_label.setStyleSheet("color: white;")
        control_row.addWidget(size_label)

        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.1, 1000.0)
        self.size_spinbox.setSingleStep(0.1)
        self.size_spinbox.setDecimals(3)
        self.size_spinbox.setValue(self.point_size)
        self.size_spinbox.valueChanged.connect(self.on_point_size_changed)
        self.size_spinbox.setMaximumWidth(70)
        control_row.addWidget(self.size_spinbox)

        # Lines checkbox
        self.lines_checkbox = QCheckBox("Lines")
        self.lines_checkbox.setStyleSheet("color: white;")
        self.lines_checkbox.setChecked(self.draw_lines)
        self.lines_checkbox.toggled.connect(self.on_lines_toggled)
        control_row.addWidget(self.lines_checkbox)

        # View buttons
        for mode in ("XY", "XZ", "YZ"):
            button = QPushButton(mode)
            button.clicked.connect(self._make_view_button_handler(mode.lower()))
            button.setMaximumWidth(40)
            control_row.addWidget(button)

        # Reset button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        reset_button.setMaximumWidth(80)
        control_row.addWidget(reset_button)

        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.immediate_exit)
        exit_button.setMaximumWidth(60)
        control_row.addWidget(exit_button)

        # Info label
        info_text = f"{len(self.points3d):,} pts"
        self.info_label = QLabel(info_text)
        self.info_label.setStyleSheet("color: white;")
        control_row.addWidget(self.info_label)

        control_row.addStretch()

        main_layout.addLayout(control_row)
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("3D Point Viewer (Matplotlib)")

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow { background-color: #353535; }
            QWidget { background-color: #353535; color: white; }
            QDoubleSpinBox { background-color: #454545; border: 1px solid #656565; }
            QPushButton { background-color: #454545; border: 1px solid #656565; padding: 4px; }
            QPushButton:hover { background-color: #565656; }
        """
        )

    def _make_colors(self):
        """Generate colors from color data or use default."""
        return self.color_manager.make_colors(self.color_data)

    def _set_equal_aspect_3d(self):
        """Set equal aspect ratio for 3D plot."""
        # Get current scaled points for bounds calculation
        scaled_points = self.points3d * self.state.scale

        if len(scaled_points) == 0:
            return

        # Calculate bounds
        mins = scaled_points.min(axis=0)
        maxs = scaled_points.max(axis=0)
        ranges = maxs - mins
        max_range = ranges.max()

        # Expand all axes to have the same range
        centers = (mins + maxs) / 2
        half_range = max_range / 2

        self.ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
        self.ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
        self.ax.set_zlim(centers[2] - half_range, centers[2] + half_range)

    def _update_plot(self):
        """Update the plot with current scaling."""
        # Apply current scaling
        scaled_points = self.points3d * self.state.scale

        # Downsample for performance if needed
        if len(scaled_points) > self.max_display_points:
            step = len(scaled_points) // self.max_display_points
            display_points = scaled_points[::step]
            if self.color_data is not None:
                display_colors = self.color_data[::step]
            else:
                display_colors = None
        else:
            display_points = scaled_points
            display_colors = self.color_data

        # Clear and replot
        self.ax.clear()

        # Reapply styling after clear
        self.ax.set_facecolor("black")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("white")
        self.ax.yaxis.pane.set_edgecolor("white")
        self.ax.zaxis.pane.set_edgecolor("white")
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        if len(display_points) > 0:
            # Plot points
            if display_colors is not None:
                color_norm = (display_colors - display_colors.min()) / max(
                    np.ptp(display_colors), 1e-6
                )
                scatter = self.ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    c=color_norm,
                    s=self.point_size,
                    cmap="viridis",
                    alpha=0.8,
                    rasterized=not self.disable_antialiasing,
                )
            else:
                self.ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    c="white",
                    s=self.point_size,
                    alpha=0.8,
                    rasterized=not self.disable_antialiasing,
                )

            # Plot lines if enabled
            if self.draw_lines and len(display_points) > 1:
                self.ax.plot(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    color="gray",
                    linewidth=1,
                    alpha=0.6,
                )

        # Set equal aspect ratio
        self._set_equal_aspect_3d()

    def _make_view_button_handler(self, mode: str):
        """Create a button handler for view mode."""

        def handler():
            self.set_view(mode)
            self.canvas.setFocus()

        return handler

    def set_view(self, mode: str):
        """Set orthographic view mode."""
        if mode == "xy":
            self.ax.view_init(elev=90, azim=0)
        elif mode == "xz":
            self.ax.view_init(elev=0, azim=0)
        elif mode == "yz":
            self.ax.view_init(elev=0, azim=90)

        self.canvas.draw_idle()
        print(f"[INFO] View set to {mode.upper()}")

    def on_matplotlib_key_press(self, event):
        """Handle matplotlib key press events."""
        if event.key == "q" or event.key == "escape":
            print(f"[INFO] '{event.key}' pressed, closing viewer.")
            QApplication.instance().quit()
            sys.exit(0)

        # Handle other keys for scaling
        elif event.key:
            key_name = event.key.upper()
            # Convert matplotlib key names to our format
            if key_name in ["X", "Y", "Z"]:
                has_shift = False  # matplotlib reports shift separately
                self.state.add_key(key_name, has_shift)

    def immediate_exit(self):
        """Immediately exit the application."""
        print("[INFO] Exit button pressed, closing viewer.")
        QApplication.instance().quit()
        sys.exit(0)

    def reset_view(self):
        """Reset view to show all points."""
        # Reset scaling
        self.state.scale[:] = 1.0
        self.state.velocity[:] = 0.0

        # Reset camera to default 3D view
        self.ax.view_init(elev=30, azim=45)

        self._update_plot()
        self.canvas.draw_idle()

    def on_acceleration_changed(self, value: float):
        """Handle acceleration change."""
        self.acceleration = value
        self.keyboard_manager.set_acceleration(value)

    def on_point_size_changed(self, value: float):
        """Handle point size change."""
        self.point_size = value
        self._update_plot()
        self.canvas.draw_idle()

    def on_lines_toggled(self, checked: bool):
        """Handle lines toggle."""
        self.draw_lines = checked
        self._update_plot()
        self.canvas.draw_idle()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        # Handle scaling keys
        if key_name:
            self.state.add_key(key_name, has_shift)

        # ESC to exit - immediate exit
        if event.key() == Qt.Key.Key_Escape:
            print("[INFO] 'ESC' pressed, closing viewer.")
            QApplication.instance().quit()
            sys.exit(0)

        # Q to quit - immediate exit
        elif event.key() == Qt.Key.Key_Q and not has_shift:
            print("[INFO] 'q' pressed, closing viewer.")
            QApplication.instance().quit()
            sys.exit(0)

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        key_name = event.text().upper() if event.text() else None
        if key_name:
            self.state.remove_key(key_name)
        super().keyReleaseEvent(event)

    def on_timer(self):
        """Timer callback for continuous updates."""
        frame_start = time()
        now = time()
        dt = now - self.last_time
        self.last_time = now

        # Update scaling based on keyboard input
        old_scale = self.state.scale.copy()
        self._update_scaling(dt)

        # Only update plot if scaling changed
        if not np.allclose(old_scale, self.state.scale):
            self._update_plot()
            self.canvas.draw_idle()

    def _update_scaling(self, dt: float):
        """Update scaling based on keyboard input."""
        # Use consolidated keyboard manager - matches exact behavior for 3D (X, Y, Z)
        self.keyboard_manager.update_scaling(dt, dimensions=3)

    def show_gui(self):
        """Show the GUI and start the application."""
        self.show()

        # Ensure canvas gets focus for key events
        self.canvas.setFocus()

        print("[INFO] Controls:")
        print("  Mouse: Rotate view (default matplotlib 3D controls)")
        print("  ESC or Q: Exit immediately")
        print("  X / Shift+X: Scale X axis")
        print("  Y / Shift+Y: Scale Y axis")
        print("  Z / Shift+Z: Scale Z axis")
        print("  View buttons: Set orthographic views")

        # Keep the application running
        if hasattr(QApplication.instance(), "exec"):
            QApplication.instance().exec()
        else:
            QApplication.instance().exec_()


if __name__ == "__main__":
    # Test with sample data
    app = QApplication(sys.argv)

    # Generate test data
    N = 10000
    t = np.linspace(0, 4 * np.pi, N)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    points = np.column_stack([x, y, z])
    colors = t  # Use parameter as color

    viewer = PointCloud3DViewerMatplotlib(
        points_xyz=points,
        color_data=colors,
        normalize=True,
    )
    viewer.show_gui()

    sys.exit(app.exec())
