#!/usr/bin/env python3
# tab-width:4

"""
Control Bar Integration Module for PointCloud2DViewerMatplotlib

This module handles the integration between the control bar UI and the viewer,
including signal connections, control synchronization, and state updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib


class ControlBarIntegration:
    """
    Manages control bar integration for the 2D matplotlib viewer.

    This class handles:
    - Signal connections between UI and viewer
    - Control state synchronization
    - Plot selector updates
    - View bounds display updates
    """

    def __init__(self, viewer: PointCloud2DViewerMatplotlib):
        """
        Initialize control bar integration.

        Args:
            viewer: Reference to the main viewer instance
        """
        self.viewer = viewer

    def connect_signals(self) -> None:
        """Connect all control bar signals to their handlers."""
        # Base signal map for standard controls
        signal_map = {
            # File operations
            "addRequested": self.viewer.event_handlers.on_add_files,
            # Plot selection and properties
            "plotChanged": self.viewer.event_handlers.on_plot_selection_changed,
            "visibilityToggled": self.viewer.event_handlers.on_visibility_toggled,
            # Rendering controls
            "accelChanged": self.viewer.event_handlers.on_acceleration_changed,
            "sizeChanged": self.viewer.event_handlers.on_point_size_changed,
            "linesToggled": self.viewer.event_handlers.on_lines_toggled,
            "paletteChanged": self.viewer.event_handlers.on_palette_changed,
            # Grid controls
            "gridSpacingChanged": self.viewer.event_handlers.on_grid_changed,
            "axesGridColorPickRequested": self.viewer.event_handlers.on_pick_axes_grid_color,
            "adcGridColorPickRequested": self.viewer.event_handlers.on_pick_grid2n_color,
            # View controls
            "resetRequested": self.viewer.event_handlers.reset_view,
            "exitRequested": self.viewer.event_handlers.immediate_exit,
            "fitViewRequested": self.viewer.event_handlers.fit_view_to_data,
            "applyViewRequested": self.viewer.event_handlers.apply_view_bounds,
            "applyOffsetRequested": self.viewer.event_handlers.apply_offset_values,
        }

        # Add secondary axis signals from the integration module
        if hasattr(self.viewer, "secondary_axis"):
            signal_map.update(self.viewer.secondary_axis.connect_signals())

        # Connect all signals
        self.viewer.control_bar_manager.connect_signals(signal_map)

    def sync_controls_to_selection(self) -> None:
        """Synchronize control values to currently selected plot."""
        if not self._has_control_bar():
            #print(
            #    "[DEBUG] control_bar_manager not yet initialized, skipping control sync"
            #)
            return

        props = self.viewer.plot_manager.get_selected_plot_properties()
        if not props:
            return

        # Sync basic plot properties
        self._sync_plot_properties(props)

        # Sync grid colors
        self._sync_grid_colors()

        # Sync secondary axis state
        if hasattr(self.viewer, "secondary_axis"):
            self.viewer.secondary_axis.sync_ui_state()

        # Update view bounds display
        self.update_view_bounds_display()

    def _sync_plot_properties(self, props: dict[str, Any]) -> None:
        """Sync plot-specific properties to controls."""
        manager = self.viewer.control_bar_manager

        manager.set_point_size(props["size"])
        manager.set_lines_checked(props["draw_lines"])
        manager.set_selected_palette(props["colormap"])
        manager.set_palette_enabled(props["has_color_data"])
        manager.set_offset(props["offset_x"], props["offset_y"])
        manager.set_visibility_checked(props["visible"])

    def _sync_grid_colors(self) -> None:
        """Sync grid color swatches."""
        manager = self.viewer.control_bar_manager

        manager.set_axes_grid_color_swatch(self.viewer.axes_grid_color)
        manager.set_adc_grid_color_swatch(self.viewer.grid_color)

    def refresh_plot_selector(self) -> None:
        """Update plot selector combobox with current plots."""
        try:
            labels = self.viewer.plot_manager.get_plot_labels()
            current_index = self.viewer.plot_manager.selected_plot_index

            if not labels:
                # Fallback if no labels available
                labels = [
                    f"Primary ({len(self.viewer.plot_manager.primary_points):,} pts)"
                ]

            self.viewer.control_bar_manager.set_plots(labels, current_index)

            # Force UI update
            combo = self.viewer.control_bar_manager.get_widget("plot_combo")
            if combo:
                combo.update()
                combo.repaint()

        except Exception as e:
            print(f"[ERROR] refresh_plot_selector failed: {e}")
            import traceback

            traceback.print_exc()

    def update_view_bounds_display(self) -> None:
        """Update the view bounds text fields with current values."""
        if not self._has_control_bar():
            #print(
            #    "[DEBUG] control_bar_manager not yet initialized, skipping view bounds update"
            #)
            return

        current_bounds = self.viewer.view_manager.get_current_bounds()
        self.viewer.control_bar_manager.set_view_bounds(
            current_bounds.xlim[0],
            current_bounds.xlim[1],
            current_bounds.ylim[0],
            current_bounds.ylim[1],
        )

    def set_initial_state(self) -> None:
        """Set initial control states after UI is created."""
        # Refresh plot selector
        self.refresh_plot_selector()

        # Sync controls to current selection
        self.sync_controls_to_selection()

        # Set acceleration value
        self.viewer.control_bar_manager.set_accel(self.viewer.acceleration)

    def update_info_text(self, text: str) -> None:
        """
        Update the info label text.

        Args:
            text: Text to display in info label
        """
        if self._has_control_bar():
            self.viewer.control_bar_manager.set_info_text(text)

    def update_point_count(self, count: int) -> None:
        """
        Update point count display.

        Args:
            count: Number of points to display
        """
        self.update_info_text(f"{count:,} pts")

    def get_plot_selector_index(self) -> int:
        """
        Get current plot selector index.

        Returns:
            Current index of plot selector combobox
        """
        if self._has_control_bar():
            combo = self.viewer.control_bar_manager.get_widget("plot_combo")
            if combo:
                return combo.currentIndex()
        return 0

    def set_plot_selector_index(self, index: int) -> None:
        """
        Set plot selector index.

        Args:
            index: Index to set in plot selector
        """
        if self._has_control_bar():
            combo = self.viewer.control_bar_manager.get_widget("plot_combo")
            if combo:
                combo.setCurrentIndex(index)

    def enable_controls(self, enabled: bool = True) -> None:
        """
        Enable or disable all controls.

        Args:
            enabled: Whether to enable controls
        """
        if self._has_control_bar():
            # Could iterate through all widgets and set enabled state
            # For now, just handle the main container
            pass

    def _has_control_bar(self) -> bool:
        """
        Check if control bar manager is initialized.

        Returns:
            True if control bar manager exists
        """
        return (
            hasattr(self.viewer, "control_bar_manager")
            and self.viewer.control_bar_manager is not None
        )

    def get_control_values(self) -> dict[str, Any]:
        """
        Get current values of all controls.

        Returns:
            Dictionary mapping control names to their current values
        """
        if not self._has_control_bar():
            return {}

        manager = self.viewer.control_bar_manager
        values = {}

        # Get values from various controls
        if combo := manager.get_widget("plot_combo"):
            values["selected_plot"] = combo.currentIndex()

        if spin := manager.get_widget("accel_spin"):
            values["acceleration"] = spin.value()

        if spin := manager.get_widget("size_spin"):
            values["point_size"] = spin.value()

        if chk := manager.get_widget("lines_chk"):
            values["connect_lines"] = chk.isChecked()

        if chk := manager.get_widget("visible_chk"):
            values["visible"] = chk.isChecked()

        if combo := manager.get_widget("palette_combo"):
            values["palette"] = combo.currentText()

        if combo := manager.get_widget("grid_combo"):
            values["grid_spacing"] = combo.currentText()

        # Get offset values
        values["offset_x"], values["offset_y"] = manager.get_offset_values()

        # Get view bounds
        xmin, xmax, ymin, ymax = manager.get_view_bounds()
        values["view_bounds"] = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

        return values
