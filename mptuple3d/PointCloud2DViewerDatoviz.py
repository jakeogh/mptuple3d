#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Tuple

import datoviz as dvz
import numpy as np

from .ColorManager import make_colors_from_scalar
from .utils import compute_bounds


class PointCloud2DViewerDatoviz:
    """
    Datoviz 2D point viewer with axes + pan/zoom.

    Args:
        points_xy: (N,2) float32/float64 array.
        color_data: optional (N,) floats for colormap or (N,4) uint8 for RGBA.
        normalize: if True (default), sets axes to data bounds (with padding).
        draw_lines: if True, draws a path through points (order given).
        size: point size in pixels (float). If None, auto-choose based on N.
    """

    def __init__(
        self,
        points_xy: np.ndarray,
        color_data: np.ndarray | None = None,
        normalize: bool = True,
        draw_lines: bool = False,
        size: float | None = None,
        title: str = "Datoviz 2D Point Viewer",
        background: str | tuple[int, int, int, int] = "black",
    ):
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must be (N,2) array")

        # Convert to float64 for Datoviz compatibility
        self.points_xy = np.asarray(points_xy, dtype=np.float64)
        self.N = self.points_xy.shape[0]
        self.draw_lines = draw_lines

        # App / figure / panel / axes
        self.app = dvz.App(background=background)
        self.figure = self.app.figure(1024, 768)
        self.panel = self.figure.panel()

        # Use consolidated bounds calculation
        if normalize:
            xlim, ylim = compute_bounds(self.points_xy, pad_ratio=0.05)
        else:
            # "Center only": still show data bounds, just no extra scale tricks
            xlim, ylim = compute_bounds(self.points_xy, pad_ratio=0.0)
        self.axes = self.panel.axes(xlim, ylim)  # enables pan/zoom

        # Prepare positions in NDC via axes.normalize
        # Extract x, y as float64 arrays for Datoviz
        x = self.points_xy[:, 0].astype(np.float64)
        y = self.points_xy[:, 1].astype(np.float64)
        position = self.axes.normalize(x, y)  # -> (N,3) float32

        # Colors - use consolidated color function
        if color_data is None:
            color = np.full((self.N, 4), 255, dtype=np.uint8)  # white
        else:
            if (
                color_data.ndim == 2
                and color_data.shape[1] == 4
                and color_data.dtype == np.uint8
            ):
                color = color_data  # already RGBA
            else:
                color = make_colors_from_scalar(
                    np.asarray(color_data).reshape(-1),
                    colormap="viridis",
                    backend="datoviz",
                )
                # Optional colorbar: nice freebie
                dmin = float(np.min(color_data))
                dmax = float(np.max(color_data))
                self.figure.colorbar(cmap="viridis", dmin=dmin, dmax=dmax)

        # Size
        if size is not None:
            psize = float(size)
        else:
            # Auto: smaller when very large N
            psize = 3.0 if self.N > 100_000 else 5.0
        self._sizes = np.full((self.N,), psize, dtype=np.float32)

        # Scatter visual
        self.scatter = self.app.point(position=position, color=color, size=self._sizes)
        self.panel.add(self.scatter)

        # Optional line/path through points
        self.path = None
        if self.draw_lines and self.N >= 2:
            # Path expects positions in NDC; reuse the same normalization
            self.path = self.app.path(
                position=position, color=(180, 180, 180, 200), linewidth=1.0
            )
            self.panel.add(self.path)

        # Keyboard helpers: X/Shift+X, Y/Shift+Y zoom like your VisPy viewer;
        # +/- to change point size; Q/Escape to quit.
        self._zoom_speed = 1.1

        @self.app.connect(self.figure)
        def on_keyboard(ev: dvz.Event):
            if ev.key_event() != "press":
                return
            k = (ev.key_name() or "").lower()
            if k in ("q", "escape"):
                self.app.destroy()
                return
            # Zoom controls: x / shift+x , y / shift+y
            if k == "x":
                self._zoom_axis(axis="x", factor=self._zoom_speed)
            elif k == "y":
                self._zoom_axis(axis="y", factor=self._zoom_speed)
            elif k == "X":  # some platforms report uppercase
                self._zoom_axis(axis="x", factor=1.0 / self._zoom_speed)
            elif k == "Y":
                self._zoom_axis(axis="y", factor=1.0 / self._zoom_speed)
            elif k in ("+", "="):
                self._resize_points(1.1)
            elif k in ("-", "_"):
                self._resize_points(1.0 / 1.1)

        # Optional FPS in title
        self._frames = 0

        # A small quality-of-life: mouse wheel zoom is enabled by axes/panzoom,
        # middle-drag pans, and you get labeled axes out of the box.

        self.title = title  # kept for symmetry with your viewer

    # ---- helpers ----

    def _zoom_axis(
        self,
        axis: str,
        factor: float,
    ):
        # Read current bounds and scale one axis around its center.
        (xmin, xmax), (ymin, ymax) = self.axes.bounds()
        if axis == "x":
            cx = 0.5 * (xmin + xmax)
            half = 0.5 * (xmax - xmin) / factor
            self.axes.xlim(cx - half, cx + half)
        elif axis == "y":
            cy = 0.5 * (ymin + ymax)
            half = 0.5 * (ymax - ymin) / factor
            self.axes.ylim(cy - half, cy + half)
        self.panel.update()

    def _resize_points(self, scale: float):
        # Maintain per-vertex size array; Datoviz updates via set_size().
        self._sizes = (self._sizes.astype(np.float32) * scale).clip(0.1, 4096)
        self.scatter.set_size(self._sizes)
        self.panel.update()

    def run(self):
        self.app.run()
        self.app.destroy()
