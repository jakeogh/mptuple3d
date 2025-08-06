#!/usr/bin/env python3
# tab-width:4
# pylint: disable=no-name-in-module

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QFileDialog


class PlotEventHandlers:
    """
    Handles events and user interactions for the 2D matplotlib viewer.
    Updated for efficient axis scaling instead of point coordinate transformation.
    """

    def __init__(self, viewer):
        """
        Initialize event handlers with reference to the main viewer.

        Args:
            viewer: The PointCloud2DViewerMatplotlib instance
        """
        self.viewer = viewer

        # Throttling for keyboard operations to prevent key repeat issues
        self.last_scale_update = 0.0
        self.scale_throttle_ms = 50  # Minimum 50ms between scale updates

    def _should_throttle_scaling(self) -> bool:
        """Check if scaling should be throttled to prevent key repeat issues."""
        now = time.time() * 1000  # Convert to milliseconds
        if now - self.last_scale_update < self.scale_throttle_ms:
            return True
        self.last_scale_update = now
        return False

    # -------- Timer and scaling --------
    def on_timer(self):
        """Timer callback for continuous updates with AXIS SCALING."""
        now = time.time()
        dt = now - self.viewer.last_time
        self.viewer.last_time = now

        old_scale = self.viewer.state.scale.copy()
        self.viewer.keyboard_manager.update_scaling(dt, dimensions=2)

        # Only update if scale changed AND we're not throttling
        if not np.allclose(old_scale, self.viewer.state.scale):
            if not self._should_throttle_scaling():
                # Only update plot if we're not currently busy
                if not self.viewer.busy_manager.is_busy:
                    with self.viewer.busy_manager.busy_operation("Scaling"):
                        # AXIS SCALING: Just apply axis scaling, no full plot update needed
                        self.viewer._apply_axis_scaling()
                        self.viewer.canvas.draw_idle()
                else:
                    # If busy, just skip this update - the scaling will accumulate
                    pass

    # -------- File operations --------
    def on_add_files(self):
        """Open QFileDialog for supported file types and append resulting plots as overlays."""
        # Use the file loader registry to handle file selection and loading
        paths = self.viewer.file_loader_registry.open_file_dialog(self.viewer)
        if not paths:
            return

        with self.viewer.busy_manager.busy_operation("Loading data files"):
            try:
                all_plots = self.viewer.file_loader_registry.load_files(paths)
            except Exception as exc:
                print(f"[ERROR] Failed during file loading: {exc}")
                return

            # Add all successfully loaded plots
            added = 0
            for arr in all_plots:
                try:
                    self.viewer.add_plot(arr)
                    added += 1
                except Exception as e:
                    print(f"[WARN] Skipped a plot due to error: {e}")

            if added:
                print(f"[INFO] Successfully added {added} plot(s) total")
            else:
                print("[INFO] No plots were successfully added")

    # -------- Plot selection and properties --------
    def on_plot_selection_changed(self, plot_index: int) -> None:
        """Retarget controls to the newly selected plot."""
        self.viewer.plot_manager.select_plot(plot_index)

    def on_acceleration_changed(self, value: float) -> None:
        """Handle acceleration change."""
        self.viewer.acceleration = float(value)
        self.viewer.keyboard_manager.set_acceleration(self.viewer.acceleration)

    def on_point_size_changed(self, value: float) -> None:
        """Handle point size change for selected plot."""
        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "size",
            value,
        )

    def on_lines_toggled(self, checked: bool) -> None:
        """Handle lines toggle for selected plot."""
        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "draw_lines",
            checked,
        )

    def on_palette_changed(self, palette_name: str):
        """Handle palette change for the selected plot."""
        if palette_name.startswith("───"):
            return

        plot_index = self.viewer.plot_manager.selected_plot_index
        self.viewer.plot_manager.set_plot_property(
            plot_index,
            "colormap",
            palette_name,
        )

        if plot_index == 0:
            self.viewer.color_manager.colormap = palette_name

    def on_visibility_toggled(self, visible: bool):
        """Handle visibility toggle for the selected plot."""
        with self.viewer.busy_manager.busy_operation("Toggling visibility"):
            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_visibility(plot_index, visible)

    # -------- Grid controls --------
    def on_grid_changed(self, grid_text: str):
        """Handle grid spacing changes."""
        with self.viewer.busy_manager.busy_operation("Updating grid"):
            if grid_text == "Off":
                self.viewer.grid_manager.set_grid_spacing(0, False)
            else:
                if "^" in grid_text:
                    power_str = grid_text.split("^")[1].split()[0]
                    power = int(power_str)
                    self.viewer.grid_manager.set_grid_spacing(power, True)
                else:
                    self.viewer.grid_manager.set_grid_spacing(0, False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_axes_grid_color(self):
        """Handle axes grid color picker."""
        new_hex = self.viewer._pick_color(self.viewer.axes_grid_color)
        if new_hex:
            self.viewer.axes_grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_axes_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    def on_pick_grid2n_color(self):
        """Handle ADC grid color picker."""
        new_hex = self.viewer._pick_color(self.viewer.grid_color)
        if new_hex:
            self.viewer.grid_color = new_hex
            self.viewer.grid_manager.set_grid_colors(
                self.viewer.grid_color, self.viewer.axes_grid_color
            )
            self.viewer.control_bar_manager.set_adc_grid_color_swatch(new_hex)
            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

    # -------- View operations --------
    def fit_view_to_data(self):
        """Fit the view to show all data points with proper scaling."""
        with self.viewer.busy_manager.busy_operation("Fitting view to data"):
            visible_data = self.viewer.plot_manager.get_visible_plots_data()

            if not visible_data:
                print("[INFO] No visible data to fit to")
                return

            try:
                all_points = []
                for _, offset_points, _, _ in visible_data:
                    # AXIS SCALING: Use original points, no scale multiplication
                    all_points.append(offset_points)

                new_bounds = self.viewer.view_manager.fit_to_data(
                    all_points, pad_ratio=0.05
                )

                # AXIS SCALING: Reset scale state when fitting to data
                self.viewer.state.scale[:2] = 1.0
                self.viewer.state.velocity[:2] = 0.0

                # Update base limits for future scaling
                self.viewer.base_xlim = new_bounds.xlim
                self.viewer.base_ylim = new_bounds.ylim

                self.viewer._update_plot()
                self.viewer.canvas.draw_idle()

                print(
                    f"[INFO] Fit view to data bounds: X({new_bounds.xlim[0]:.3f}, {new_bounds.xlim[1]:.3f}), Y({new_bounds.ylim[0]:.3f}, {new_bounds.ylim[1]:.3f})"
                )

            except Exception as e:
                print(f"[ERROR] Failed to fit view to data: {e}")

    def apply_view_bounds(self):
        """Apply custom view bounds from the text fields."""
        with self.viewer.busy_manager.busy_operation("Applying view bounds"):
            try:
                xmin, xmax, ymin, ymax = (
                    self.viewer.control_bar_manager.get_view_bounds()
                )

                is_valid, error_msg, new_bounds = (
                    self.viewer.view_manager.validate_bounds(
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                    )
                )

                if not is_valid:
                    print(f"[ERROR] {error_msg}")
                    return

                applied_bounds = self.viewer.view_manager.apply_validated_bounds(
                    new_bounds
                )

                # AXIS SCALING: Reset scale state when applying custom bounds
                self.viewer.state.scale[:2] = 1.0
                self.viewer.state.velocity[:2] = 0.0

                # Update base limits for future scaling
                self.viewer.base_xlim = applied_bounds.xlim
                self.viewer.base_ylim = applied_bounds.ylim

                self.viewer._update_plot()
                self.viewer.canvas.draw_idle()

                print(
                    f"[INFO] Applied custom view bounds: X({applied_bounds.xlim[0]:.3f}, {applied_bounds.xlim[1]:.3f}), Y({applied_bounds.ylim[0]:.3f}, {applied_bounds.ylim[1]:.3f})"
                )

            except Exception as e:
                print(f"[ERROR] Failed to apply view bounds: {e}")

    def apply_offset_values(self):
        """Apply offset values from spinboxes to the selected plot."""
        with self.viewer.busy_manager.busy_operation("Applying plot offset"):
            x_offset, y_offset = self.viewer.control_bar_manager.get_offset_values()

            plot_index = self.viewer.plot_manager.selected_plot_index
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "offset_x",
                x_offset,
            )
            self.viewer.plot_manager.set_plot_property(
                plot_index,
                "offset_y",
                y_offset,
            )

            plot_info = self.viewer.plot_manager.get_plot_info(plot_index)
            if plot_info:
                plot_type = (
                    "primary plot" if plot_info.is_primary else f"overlay {plot_index}"
                )
                print(
                    f"[INFO] Applied offset to {plot_type}: ({x_offset:.3f}, {y_offset:.3f})"
                )

    def reset_view(self) -> None:
        """Reset axes limits to data bounds and reset scaling."""
        with self.viewer.busy_manager.busy_operation("Resetting view"):
            # Get ALL visible plots data (not just primary!)
            visible_data = self.viewer.plot_manager.get_visible_plots_data()

            if not visible_data:
                # No visible data, just set default bounds
                new_bounds = self.viewer.view_manager.set_view_bounds(
                    xlim=(0, 1), ylim=(0, 1)
                )
            else:
                # Collect all visible points
                all_points = []
                for _, offset_points, _, _ in visible_data:
                    all_points.append(offset_points)

                pad_ratio = (
                    0.0 if (self.viewer.normalize or self.viewer.center) else 0.1
                )

                try:
                    new_bounds = self.viewer.view_manager.reset_to_data_bounds(
                        all_points, pad_ratio
                    )
                except Exception:
                    # Fallback if reset fails
                    new_bounds = self.viewer.view_manager.fit_to_data(
                        all_points, pad_ratio
                    )

            # AXIS SCALING: Reset scale state completely
            self.viewer.state.scale[:] = 1.0
            self.viewer.state.velocity[:] = 0.0

            # Update base limits for future scaling
            self.viewer.base_xlim = new_bounds.xlim
            self.viewer.base_ylim = new_bounds.ylim

            self.viewer.view_manager.set_zoom_box_active(False)

            self.viewer._update_plot()
            self.viewer.canvas.draw_idle()

            # Remove the unnecessary _update_view_bounds_display() call

    def immediate_exit(self) -> None:
        """Close the viewer window immediately."""
        print("[INFO] Exit button pressed, closing viewer.")
        self.viewer.close()
