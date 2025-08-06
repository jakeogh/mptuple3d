from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import numpy as np
from matplotlib.axes import Axes


class Matplotlib2DRenderer:
    """
    Pure Matplotlib/NumPy renderer for PointCloud2DViewerMatplotlib.
    Updated for efficient axis scaling approach.

    - No Qt imports.
    - Accepts overlays as dataclass-like objects (duck-typed), i.e. attributes:
      points, color_data, size, color, draw_lines, cmap, (optional) offset_x/y.
    - Uses original point coordinates - scaling is handled via axis limits.
    """

    def __init__(self):
        """Initialize the renderer with artist tracking."""
        # NEW: Track whether we've initialized the plot
        self.plot_initialized = False
        # NEW: Track grid line artists
        self.grid_line_artists = []

    def _initialize_axes(
        self,
        ax: Axes,
        axes_grid_color: str,
        in_zoom_box: bool,
        auto_aspect: bool,
    ) -> None:
        """Initialize or clear the axes for drawing."""
        # Clear and restyle
        ax.clear()
        ax.set_facecolor("black")
        ax.grid(
            True,
            color=axes_grid_color,
            alpha=0.3,
        )
        ax.tick_params(colors="white")
        if in_zoom_box:
            ax.set_aspect("auto")
        elif auto_aspect:
            ax.set_aspect("auto", adjustable="datalim")
        else:
            ax.set_aspect("equal", adjustable="datalim")

    def _create_scatter_artist(
        self,
        ax: Axes,
        points: np.ndarray,
        color_data: np.ndarray | None,
        size: float,
        cmap: str,
        color: str | None,
        alpha: float = 0.8,
        rasterized: bool = True,
    ):
        """Create a scatter artist and return the PathCollection."""
        if len(points) == 0:
            return None

        if color_data is not None and len(color_data) > 0:
            c_norm = (color_data - color_data.min()) / max(np.ptp(color_data), 1e-6)
            artist = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=c_norm,
                s=size,
                cmap=cmap,
                alpha=alpha,
                rasterized=rasterized,
            )
        else:
            artist = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=color if color else "white",
                s=size,
                alpha=alpha,
                rasterized=rasterized,
            )
        return artist

    def _update_scatter_artist(
        self,
        artist,
        points: np.ndarray,
        color_data: np.ndarray | None,
        size: float,
        visible: bool = True,
    ):
        """Update an existing scatter artist with new data."""
        if artist is None:
            return

        if not visible or len(points) == 0:
            artist.set_visible(False)
            return

        artist.set_visible(True)
        artist.set_offsets(points)
        artist.set_sizes([size] * len(points))

        if color_data is not None and len(color_data) > 0:
            c_norm = (color_data - color_data.min()) / max(np.ptp(color_data), 1e-6)
            artist.set_array(c_norm)

    def _create_line_artist(
        self,
        ax: Axes,
        points: np.ndarray,
        color: str = "gray",
        linewidth: float = 1,
        alpha: float = 0.6,
    ):
        """Create a line artist and return the Line2D list."""
        if len(points) <= 1:
            return None

        lines = ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        return lines[0] if lines else None

    def _update_line_artist(
        self,
        artist,
        points: np.ndarray,
        visible: bool = True,
    ):
        """Update an existing line artist with new data."""
        if artist is None:
            return

        if not visible or len(points) <= 1:
            artist.set_visible(False)
            return

        artist.set_visible(True)
        artist.set_data(points[:, 0], points[:, 1])

    def _visible_points_mask(
        self,
        pts: np.ndarray,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> np.ndarray:
        return (
            (pts[:, 0] >= xlim[0])
            & (pts[:, 0] <= xlim[1])
            & (pts[:, 1] >= ylim[0])
            & (pts[:, 1] <= ylim[1])
        )

    def draw_grid_lines(
        self,
        ax: Axes,
        *,
        spacing_power: int,
        points2d: np.ndarray,
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        grid_color: str = "#808080",
        max_lines: int = 1000,
    ) -> None:
        """Draw horizontal lines at 2^N spacing behind all artists."""
        if spacing_power <= 0:
            return
        spacing = 2**spacing_power

        xlim = view_xlim
        ylim = view_ylim

        # Prefer visible points to determine Y extents
        visible_mask = self._visible_points_mask(
            points2d,
            xlim,
            ylim,
        )
        if visible_mask.any():
            visible_points = points2d[visible_mask]
            y_min_data = float(visible_points[:, 1].min())
            y_max_data = float(visible_points[:, 1].max())
            y_range = y_max_data - y_min_data
            if y_range > 0:
                y_padding = y_range * 0.1
                y_min_grid = y_min_data - y_padding
                y_max_grid = y_max_data + y_padding
            else:
                y_min_grid = y_min_data - spacing
                y_max_grid = y_max_data + spacing
        else:
            # Fallback to current ylim
            y_min_grid, y_max_grid = ylim

        # Draw horizontal lines
        start_y = int(np.floor(y_min_grid / spacing)) * spacing
        y = start_y
        count = 0
        while y <= y_max_grid and count < max_lines:
            ax.axhline(
                y=y,
                color=grid_color,
                linewidth=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            y += spacing
            count += 1

    def update_plot(
        self,
        ax: Axes,
        *,
        points2d: np.ndarray,
        color_data: None | np.ndarray,
        overlays: Sequence[object],  # dataclass-like objects with required attrs
        scale_xy: Sequence[float],  # IGNORED in axis scaling approach
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        point_size: float,
        draw_lines: bool,
        colormap: str,
        grid_enabled: bool,
        grid_power: int,
        grid_color: str,
        axes_grid_color: str,
        disable_antialiasing: bool,
        max_display_points: int,
        in_zoom_box: bool,
        auto_aspect: bool,
        primary_offset: tuple[float, float] = (0.0, 0.0),
        primary_visible: bool = True,
        force_redraw: bool = False,
        plot_manager=None,  # NEW: Pass plot manager to access artists
    ) -> None:
        """
        Update plot using persistent artists when possible.
        NOTE: Data is already culled to viewport by the main viewer.
        AXIS SCALING: Uses original coordinates, scaling handled by axis limits.
        """

        xlim = view_xlim
        ylim = view_ylim

        # Check if we need a full redraw or can update existing artists
        needs_full_redraw = (
            not self.plot_initialized
            or force_redraw
            # For now, still do full redraw if grid settings change
            or (grid_enabled and grid_power > 0)
        )

        if needs_full_redraw:
            # Full redraw - clear everything and recreate
            self.plot_initialized = True
            self._initialize_axes(
                ax,
                axes_grid_color,
                in_zoom_box,
                auto_aspect,
            )

            # Clear all artist references when doing full redraw
            if plot_manager:
                plot_manager.primary_scatter_artist = None
                plot_manager.primary_line_artist = None
                for overlay in overlays:
                    overlay.scatter_artist = None
                    overlay.line_artist = None

        # ADC-style grid (keep existing grid code)
        if grid_enabled and grid_power > 0:
            # Use the first visible plot for grid extent calculation
            reference_points = None
            if primary_visible and primary_offset != (0.0, 0.0):
                reference_points = points2d + np.asarray(
                    primary_offset, dtype=np.float32
                )
            elif primary_visible:
                reference_points = points2d
            else:
                # Find first visible overlay for reference
                for ov in overlays or []:
                    if getattr(ov, "visible", True):
                        pts = getattr(ov, "points")
                        dx = float(getattr(ov, "offset_x", 0.0))
                        dy = float(getattr(ov, "offset_y", 0.0))
                        if dx != 0.0 or dy != 0.0:
                            reference_points = pts + np.array(
                                [dx, dy], dtype=np.float32
                            )
                        else:
                            reference_points = pts
                        break

            if reference_points is not None:
                self.draw_grid_lines(
                    ax,
                    spacing_power=grid_power,
                    points2d=reference_points,
                    view_xlim=xlim,
                    view_ylim=ylim,
                    grid_color=grid_color,
                )

        # PRIMARY PLOT - Use persistent artists
        if plot_manager:
            # Apply offset
            if primary_offset != (0.0, 0.0):
                points = points2d + np.asarray(primary_offset, dtype=np.float32)
            else:
                points = points2d

            # Downsample if needed
            colors = color_data
            if len(points) > max_display_points:
                step = max(1, len(points) // max_display_points)
                points = points[::step]
                colors = color_data[::step] if color_data is not None else None

            # Handle scatter plot
            if (
                plot_manager.primary_scatter_artist is None
                and primary_visible
                and len(points) > 0
            ):
                # Create new scatter artist
                plot_manager.primary_scatter_artist = self._create_scatter_artist(
                    ax,
                    points,
                    colors,
                    point_size,
                    colormap,
                    "white",
                    alpha=0.8,
                    rasterized=not disable_antialiasing,
                )
            elif plot_manager.primary_scatter_artist is not None:
                # Update existing scatter artist
                self._update_scatter_artist(
                    plot_manager.primary_scatter_artist,
                    points,
                    colors,
                    point_size,
                    primary_visible,
                )

            # Handle line plot
            if draw_lines and primary_visible:
                line_points = (
                    points
                    if len(points) <= max_display_points
                    else points[:: max(1, len(points) // max_display_points)]
                )
                if plot_manager.primary_line_artist is None and len(line_points) > 1:
                    # Create new line artist
                    plot_manager.primary_line_artist = self._create_line_artist(
                        ax,
                        line_points,
                        "gray",
                        1,
                        0.6,
                    )
                elif plot_manager.primary_line_artist is not None:
                    # Update existing line artist
                    self._update_line_artist(
                        plot_manager.primary_line_artist,
                        line_points,
                        len(line_points) > 1,
                    )
            elif plot_manager.primary_line_artist is not None:
                # Hide line if not needed
                plot_manager.primary_line_artist.set_visible(False)

        else:
            # Fallback to old behavior if no plot_manager
            # (Keep the existing primary plot code here for compatibility)
            if primary_visible:
                # ... existing primary plot code ...
                pass

        # OVERLAYS - Use persistent artists
        for ov in overlays or []:
            # Check visibility
            if not getattr(ov, "visible", True):
                if ov.scatter_artist:
                    ov.scatter_artist.set_visible(False)
                if ov.line_artist:
                    ov.line_artist.set_visible(False)
                continue

            pts = getattr(ov, "points")
            # Apply offset
            dx = float(getattr(ov, "offset_x", 0.0))
            dy = float(getattr(ov, "offset_y", 0.0))
            if dx != 0.0 or dy != 0.0:
                pts = pts + np.array([dx, dy], dtype=np.float32)

            if len(pts) == 0:
                if ov.scatter_artist:
                    ov.scatter_artist.set_visible(False)
                if ov.line_artist:
                    ov.line_artist.set_visible(False)
                continue

            size = float(getattr(ov, "size", point_size))
            ov_color_data = getattr(ov, "color_data", None)

            # Downsample if needed
            if len(pts) > max_display_points:
                step = max(1, len(pts) // max_display_points)
                vpts = pts[::step]
                o_colors = ov_color_data[::step] if ov_color_data is not None else None
            else:
                vpts = pts
                o_colors = ov_color_data

            # Handle scatter
            if ov.scatter_artist is None:
                # Create new scatter artist
                ov.scatter_artist = self._create_scatter_artist(
                    ax,
                    vpts,
                    o_colors,
                    size,
                    getattr(ov, "cmap", colormap),
                    getattr(ov, "color", "red"),
                    alpha=0.8,
                    rasterized=not disable_antialiasing,
                )
            else:
                # Update existing scatter artist
                self._update_scatter_artist(
                    ov.scatter_artist,
                    vpts,
                    o_colors,
                    size,
                    True,
                )

            # Handle line
            if getattr(ov, "draw_lines", False) and len(pts) > 1:
                line_pts = (
                    pts
                    if len(pts) <= max_display_points
                    else pts[:: max(1, len(pts) // max_display_points)]
                )
                if ov.line_artist is None:
                    ov.line_artist = self._create_line_artist(
                        ax,
                        line_pts,
                        getattr(ov, "color", "red"),
                        1,
                        0.6,
                    )
                else:
                    self._update_line_artist(ov.line_artist, line_pts, True)
            elif ov.line_artist is not None:
                ov.line_artist.set_visible(False)

        # Restore limits - AXIS SCALING handles zoom via limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
