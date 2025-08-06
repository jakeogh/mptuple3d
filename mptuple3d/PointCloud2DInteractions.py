from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent


class PointCloud2DInteractions:
    """Encapsulates Matplotlib + Qt interaction handlers.

    The viewer delegates UI events to this object; state changes and rendering
    are still performed via the viewer's public/internal methods.
    """

    def __init__(
        self,
        viewer,
        ax: Axes,
        canvas: FigureCanvasQTAgg,
        state,
    ):
        self.viewer = viewer
        self.ax = ax
        self.canvas = canvas
        self.state = state

    # ---------- Matplotlib callbacks ----------
    def on_zoom_box(
        self,
        eclick,
        erelease,
    ):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        self.viewer._in_zoom_box = True
        self.ax.set_aspect("auto")
        self.viewer.current_xlim = (x_min, x_max)
        self.viewer.current_ylim = (y_min, y_max)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.viewer._update_plot()
        self.canvas.draw_idle()

    def on_mouse_scroll(self, event):
        if event.inaxes != self.ax:
            return
        x_mouse, y_mouse = event.xdata, event.ydata
        if x_mouse is None or y_mouse is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zoom_factor = 1.1 if event.step > 0 else 1.0 / 1.1
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor
        x_center_offset = x_mouse - (xlim[0] + xlim[1]) / 2
        y_center_offset = y_mouse - (ylim[0] + ylim[1]) / 2
        new_x_center = (xlim[0] + xlim[1]) / 2 + x_center_offset * (1 - 1 / zoom_factor)
        new_y_center = (ylim[0] + ylim[1]) / 2 + y_center_offset * (1 - 1 / zoom_factor)
        new_xlim: tuple[float, float] = (
            new_x_center - new_x_range / 2,
            new_x_center + new_x_range / 2,
        )
        new_ylim: tuple[float, float] = (
            new_y_center - new_y_range / 2,
            new_y_center + new_y_range / 2,
        )
        self.viewer.current_xlim = new_xlim
        self.viewer.current_ylim = new_ylim
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self.viewer._update_plot()
        self.canvas.draw_idle()

    def on_matplotlib_key_press(self, event):
        if event.key == "q" or event.key == "escape":
            print(f"[INFO] '{event.key}' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            # QApplication.instance().quit()
            # sys.exit(0)
            self.viewer.close()
        elif event.key:
            key_name = event.key.upper()
            if key_name in ["X", "Y", "Z"]:
                self.state.add_key(key_name, has_shift=False)

    # ---------- Qt key events (delegated) ----------
    def keyPressEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key_name:
            self.state.add_key(key_name, has_shift)
        if event.key() == Qt.Key.Key_Escape:
            print("[INFO] 'ESC' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            # QApplication.instance().quit()
            # sys.exit(0)
            self.viewer.close()
        elif event.key() == Qt.Key.Key_Q:
            print("[INFO] 'q' pressed, closing viewer.")
            from PyQt6.QtWidgets import QApplication

            # QApplication.instance().quit()
            # sys.exit(0)
            self.viewer.close()

    def keyReleaseEvent(self, event: QKeyEvent):
        key_name = event.text().upper() if event.text() else None
        if key_name:
            self.state.remove_key(key_name)
