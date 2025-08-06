#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal

from .CoordinateTransformEngine import CoordinateTransformEngine
from .CoordinateTransformEngine import TransformParams
from .PointCloud2DMatplotlibOverlay import Overlay

if TYPE_CHECKING:
    pass


@dataclass
class PlotInfo:
    """Information about a single plot."""

    index: int
    name: str
    point_count: int
    is_primary: bool
    visible: bool
    size: float
    colormap: str
    has_color_data: bool
    offset_x: float
    offset_y: float
    draw_lines: bool


class PlotDataManagerSignals(QObject):
    """Signal hub for plot data events."""

    # Plot structure changes
    plotAdded = pyqtSignal(int)  # plot_index
    plotRemoved = pyqtSignal(int)  # plot_index
    plotsChanged = pyqtSignal()  # general plot list change

    # Plot selection changes
    selectionChanged = pyqtSignal(int)  # new_plot_index

    # Plot property changes
    plotVisibilityChanged = pyqtSignal(int, bool)  # plot_index, visible
    plotPropertiesChanged = pyqtSignal(int)  # plot_index


class PlotDataManager:
    """
    Manages multiple plot data and metadata for PointCloud2DViewerMatplotlib.

    Handles:
    - Plot creation, deletion, and management
    - Plot selection and switching
    - Plot visibility and properties
    - Coordinate transformation coordination
    - Plot metadata and statistics
    """

    def __init__(
        self,
        primary_points: np.ndarray,
        primary_color_data: None | np.ndarray,
        primary_colormap: str,
        primary_transform_params: TransformParams,
        transform_engine: CoordinateTransformEngine,
    ):
        """
        Initialize plot data manager with primary plot.

        Args:
            primary_points: Primary plot points (already transformed)
            primary_color_data: Primary plot color data (optional)
            primary_transform_params: Transformation parameters used for primary
            transform_engine: Coordinate transformation engine
        """
        self.signals = PlotDataManagerSignals()
        self.transform_engine = transform_engine

        # Store primary plot data
        self.primary_points = primary_points
        self.primary_color_data = primary_color_data
        self.primary_transform_params = primary_transform_params
        self.primary_visible = True

        # Primary plot properties
        self.primary_size = 2.0
        self.primary_colormap = primary_colormap
        self.primary_draw_lines = False
        self.primary_offset_x = 0.0
        self.primary_offset_y = 0.0

        # NEW: Artist tracking for primary plot
        self.primary_scatter_artist = None
        self.primary_line_artist = None

        # Overlay storage
        self.overlays: list[Overlay] = []

        # Selection state
        self.selected_plot_index = 0  # 0 = primary, 1+ = overlays

        # Plot counter for naming
        self.overlay_counter = 0

        # NEW: Cache for plot labels
        self._cached_labels = None
        self._invalidate_label_cache()  # Initialize the cache

    def _invalidate_label_cache(self):
        """Mark label cache as needing rebuild."""
        self._cached_labels = None

    def _rebuild_label_cache(self):
        """Rebuild the label cache - O(n) but only when needed."""
        self._cached_labels = []
        # Primary
        self._cached_labels.append(f"Primary ({len(self.primary_points):,} pts)")
        # Overlays
        for i, overlay in enumerate(self.overlays):
            self._cached_labels.append(f"Overlay {i + 1} ({len(overlay.points):,} pts)")

    def get_plot_count(self) -> int:
        """Get total number of plots (primary + overlays)."""
        return 1 + len(self.overlays)

    def get_overlay_count(self) -> int:
        """Get number of overlay plots."""
        return len(self.overlays)

    def is_primary_selected(self) -> bool:
        """Check if primary plot is currently selected."""
        return self.selected_plot_index == 0

    def get_selected_overlay(self) -> None | Overlay:
        """Get currently selected overlay, or None if primary is selected."""
        if self.is_primary_selected():
            return None

        overlay_index = self.selected_plot_index - 1
        if 0 <= overlay_index < len(self.overlays):
            return self.overlays[overlay_index]

        return None

    def select_plot(self, plot_index: int) -> bool:
        """
        Select a plot by index.

        Args:
            plot_index: 0 for primary, 1+ for overlays

        Returns:
            True if selection changed, False if invalid or same
        """
        if plot_index < 0 or plot_index >= self.get_plot_count():
            return False

        if plot_index != self.selected_plot_index:
            self.selected_plot_index = plot_index
            self.signals.selectionChanged.emit(plot_index)
            return True

        return False

    def get_plot_info(self, plot_index: int) -> None | PlotInfo:
        """
        Get plot information by index.

        Args:
            plot_index: 0 for primary, 1+ for overlays

        Returns:
            PlotInfo object or None if invalid index
        """
        if plot_index == 0:
            # Primary plot
            return PlotInfo(
                index=0,
                name=f"Primary ({len(self.primary_points):,} pts)",
                point_count=len(self.primary_points),
                is_primary=True,
                visible=self.primary_visible,
                size=self.primary_size,
                colormap=self.primary_colormap,
                has_color_data=self.primary_color_data is not None,
                offset_x=self.primary_offset_x,
                offset_y=self.primary_offset_y,
                draw_lines=self.primary_draw_lines,
            )

        overlay_index = plot_index - 1
        if 0 <= overlay_index < len(self.overlays):
            overlay = self.overlays[overlay_index]
            # Fixed: Better handling of overlay visibility attribute
            overlay_visible = getattr(
                overlay,
                "visible",
                True,
            )
            # Ensure it's a boolean
            if not isinstance(overlay_visible, bool):
                overlay_visible = True

            return PlotInfo(
                index=plot_index,
                name=f"Overlay {overlay_index + 1} ({len(overlay.points):,} pts)",
                point_count=len(overlay.points),
                is_primary=False,
                visible=overlay_visible,
                size=overlay.size,
                colormap=overlay.cmap,
                has_color_data=overlay.color_data is not None,
                offset_x=overlay.offset_x,
                offset_y=overlay.offset_y,
                draw_lines=overlay.draw_lines,
            )

        return None

    def get_plot_labels(self) -> list[str]:
        """Get list of plot labels for UI display - CACHED."""
        if self._cached_labels is None:
            self._rebuild_label_cache()
        return (
            self._cached_labels.copy()
        )  # Return a copy to prevent external modification

    def add_overlay(
        self,
        points_xyz: np.ndarray,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        colormap: str | None = None,
        size: float | None = None,
        draw_lines: bool | None = None,
        transform_params: dict | None = None,
    ) -> tuple[int, dict]:
        """
        Add an overlay plot.

        Args:
            points_xyz: Array with at least 2 columns (X,Y) and optional color data
            x_offset: X-axis offset for the overlay plot
            y_offset: Y-axis offset for the overlay plot
            colormap: Colormap name (None = use primary plot's colormap)
            size: Point size (None = use primary plot's size)
            draw_lines: Whether to connect points (None = use primary setting)
            transform_params: Dict with transformation parameters

        Returns:
            Tuple of (overlay_index, transform_params_dict)
        """
        if not isinstance(points_xyz, np.ndarray):
            raise TypeError("points_xyz must be a numpy.ndarray")
        if points_xyz.ndim != 2 or points_xyz.shape[1] < 2:
            raise ValueError("points_xyz must be 2D with at least 2 columns (X,Y).")

        # Extract points and color data
        pts = points_xyz[:, :2].astype(np.float32)
        cdata = (
            points_xyz[:, 2].astype(np.float32) if points_xyz.shape[1] >= 3 else None
        )

        # Apply coordinate transformation
        if transform_params is not None:
            # Use provided transformation parameters
            transform_params_obj = TransformParams.from_dict(transform_params)
            pts = self.transform_engine.apply_transform(pts, transform_params_obj)
            result_transform_params = transform_params.copy()
        else:
            # Use the same transformation as the primary plot
            pts = self.transform_engine.apply_transform(
                pts, self.primary_transform_params
            )
            result_transform_params = self.primary_transform_params.to_dict()

        # Use defaults from primary plot if not specified
        overlay_colormap = colormap if colormap is not None else self.primary_colormap
        overlay_size = size if size is not None else self.primary_size
        overlay_connect = (
            draw_lines
            if draw_lines is not None
            else self.primary_draw_lines
        )

        # Create overlay - ensure visible is explicitly set
        self.overlay_counter += 1
        overlay = Overlay(
            points=pts,
            color_data=cdata,
            draw_lines=overlay_connect,
            size=overlay_size,
            color=None,  # Let color be handled by color_data
            cmap=overlay_colormap,
            offset_x=float(x_offset),
            offset_y=float(y_offset),
            visible=True,  # Explicitly set visible=True for new overlays
        )

        self.overlays.append(overlay)
        overlay_index = len(self.overlays)  # 1-based for UI

        # NEW: Efficiently update cache by just appending
        if self._cached_labels is not None:
            label = f"Overlay {overlay_index} ({len(overlay.points):,} pts)"
            self._cached_labels.append(label)

        # Emit signals
        self.signals.plotAdded.emit(overlay_index)
        # Don't emit plotsChanged here - let the caller decide if a full update is needed

        return overlay_index, result_transform_params

    def remove_overlay(self, overlay_index: int) -> bool:
        """
        Remove an overlay by index.

        Args:
            overlay_index: 1-based overlay index (1 = first overlay)

        Returns:
            True if removed, False if invalid index
        """
        array_index = overlay_index - 1
        if 0 <= array_index < len(self.overlays):
            self.overlays.pop(array_index)

            # NEW: Invalidate cache when removing (requires rebuild)
            self._invalidate_label_cache()

            # Adjust selection if necessary
            if self.selected_plot_index > overlay_index:
                self.selected_plot_index -= 1
            elif self.selected_plot_index == overlay_index:
                # Select primary if we removed the selected overlay
                self.selected_plot_index = 0
                self.signals.selectionChanged.emit(0)

            # Emit signals
            self.signals.plotRemoved.emit(overlay_index)
            self.signals.plotsChanged.emit()

            return True

        return False

    def clear_overlays(self) -> int:
        """
        Remove all overlays.

        Returns:
            Number of overlays removed
        """
        count = len(self.overlays)
        self.overlays.clear()

        # NEW: Invalidate cache when clearing
        self._invalidate_label_cache()

        # Reset selection to primary
        if self.selected_plot_index > 0:
            self.selected_plot_index = 0
            self.signals.selectionChanged.emit(0)

        if count > 0:
            self.signals.plotsChanged.emit()

        return count

    def set_plot_visibility(
        self,
        plot_index: int,
        visible: bool,
    ) -> bool:
        """
        Set plot visibility.

        Args:
            plot_index: 0 for primary, 1+ for overlays
            visible: Whether plot should be visible

        Returns:
            True if changed, False if invalid index or no change
        """
        if plot_index == 0:
            if self.primary_visible != visible:
                self.primary_visible = visible
                self.signals.plotVisibilityChanged.emit(plot_index, visible)
                return True
        else:
            overlay_index = plot_index - 1
            if 0 <= overlay_index < len(self.overlays):
                overlay = self.overlays[overlay_index]
                current_visibility = getattr(
                    overlay,
                    "visible",
                    True,
                )
                if current_visibility != visible:
                    overlay.visible = visible
                    self.signals.plotVisibilityChanged.emit(plot_index, visible)
                    return True

        return False

    def set_plot_property(
        self,
        plot_index: int,
        property_name: str,
        value: Any,
    ) -> bool:
        """
        Set a plot property.

        Args:
            plot_index: 0 for primary, 1+ for overlays
            property_name: Name of property to set
            value: New value

        Returns:
            True if changed, False if invalid
        """
        changed = False

        if plot_index == 0:
            # Primary plot properties
            if property_name == "size" and self.primary_size != value:
                self.primary_size = float(value)
                changed = True
            elif property_name == "colormap" and self.primary_colormap != value:
                self.primary_colormap = str(value)
                changed = True
            elif (
                property_name == "draw_lines"
                and self.primary_draw_lines != value
            ):
                self.primary_draw_lines = bool(value)
                changed = True
            elif property_name == "offset_x" and self.primary_offset_x != value:
                self.primary_offset_x = float(value)
                changed = True
            elif property_name == "offset_y" and self.primary_offset_y != value:
                self.primary_offset_y = float(value)
                changed = True
        else:
            # Overlay properties
            overlay_index = plot_index - 1
            if 0 <= overlay_index < len(self.overlays):
                overlay = self.overlays[overlay_index]

                if property_name == "size" and overlay.size != value:
                    overlay.size = float(value)
                    changed = True
                elif property_name == "colormap" and overlay.cmap != value:
                    overlay.cmap = str(value)
                    changed = True
                elif (
                    property_name == "draw_lines"
                    and overlay.draw_lines != value
                ):
                    overlay.draw_lines = bool(value)
                    changed = True
                elif property_name == "offset_x" and overlay.offset_x != value:
                    overlay.offset_x = float(value)
                    changed = True
                elif property_name == "offset_y" and overlay.offset_y != value:
                    overlay.offset_y = float(value)
                    changed = True

        if changed:
            self.signals.plotPropertiesChanged.emit(plot_index)

        return changed

    def get_visible_plots_data(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Get data for all visible plots with their offsets applied.

        Returns:
            List of (points, offsets_applied_points, offset_x, offset_y) tuples
        """
        visible_data = []

        # Add primary plot if visible
        if self.primary_visible:
            primary_with_offset = self.primary_points + np.array(
                [self.primary_offset_x, self.primary_offset_y]
            )
            visible_data.append(
                (
                    self.primary_points,
                    primary_with_offset,
                    self.primary_offset_x,
                    self.primary_offset_y,
                )
            )

        # Add visible overlays
        for overlay in self.overlays:
            overlay_visible = getattr(
                overlay,
                "visible",
                True,
            )
            if overlay_visible:
                overlay_with_offset = overlay.points + np.array(
                    [overlay.offset_x, overlay.offset_y]
                )
                visible_data.append(
                    (
                        overlay.points,
                        overlay_with_offset,
                        overlay.offset_x,
                        overlay.offset_y,
                    )
                )

        return visible_data

    def get_all_overlays(self) -> list[Overlay]:
        """Get list of all overlay objects."""
        return self.overlays.copy()

    def get_primary_data(self) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """
        Get primary plot data.

        Returns:
            Tuple of (points, color_data, visible)
        """
        return self.primary_points, self.primary_color_data, self.primary_visible

    def get_primary_properties(self) -> dict[str, Any]:
        """Get primary plot properties as dictionary."""
        return {
            "size": self.primary_size,
            "colormap": self.primary_colormap,
            "draw_lines": self.primary_draw_lines,
            "offset_x": self.primary_offset_x,
            "offset_y": self.primary_offset_y,
            "visible": self.primary_visible,
            "has_color_data": self.primary_color_data is not None,
        }

    def get_selected_plot_properties(self) -> dict[str, Any] | None:
        """Get properties of currently selected plot."""
        if self.is_primary_selected():
            return self.get_primary_properties()

        overlay = self.get_selected_overlay()
        if overlay:
            overlay_visible = getattr(
                overlay,
                "visible",
                True,
            )
            return {
                "size": overlay.size,
                "colormap": overlay.cmap,
                "draw_lines": overlay.draw_lines,
                "offset_x": overlay.offset_x,
                "offset_y": overlay.offset_y,
                "visible": overlay_visible,
                "has_color_data": overlay.color_data is not None,
            }

        return None

    def update_primary_properties(self, **kwargs):
        """Update multiple primary plot properties at once."""
        changed = False

        for key, value in kwargs.items():
            if self.set_plot_property(
                0,
                key,
                value,
            ):
                changed = True

        return changed

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about all plots."""
        total_points = len(self.primary_points)
        visible_points = len(self.primary_points) if self.primary_visible else 0

        visible_overlay_count = 0
        for overlay in self.overlays:
            total_points += len(overlay.points)
            overlay_visible = getattr(
                overlay,
                "visible",
                True,
            )
            if overlay_visible:
                visible_points += len(overlay.points)
                visible_overlay_count += 1

        return {
            "total_plots": self.get_plot_count(),
            "overlay_count": self.get_overlay_count(),
            "total_points": total_points,
            "visible_points": visible_points,
            "primary_visible": self.primary_visible,
            "visible_overlays": visible_overlay_count,
            "selected_plot_index": self.selected_plot_index,
            "selected_is_primary": self.is_primary_selected(),
        }
