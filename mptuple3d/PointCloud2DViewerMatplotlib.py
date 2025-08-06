#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

# pylint: disable=no-name-in-module
import sys
from time import time

import numpy as np
from asserttool import icp
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QColorDialog
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .BusyIndicatorManager import BusyIndicatorManager
from .color_paletts import COLOR_PALETTES
from .ColorManager import ColorManager
from .ControlBarIntegration import ControlBarIntegration
from .ControlBarManager import ControlBarManager
from .CoordinateTransformEngine import CoordinateTransformEngine
from .CoordinateTransformEngine import TransformParams
from .FileLoaderRegistry import FileLoaderRegistry
from .GridManager import GridManager
from .InputState import InputState
from .PlotDataManager import PlotDataManager
from .PlotEventHandlers import PlotEventHandlers
from .PointCloud2DInteractions import PointCloud2DInteractions
from .PointCloud2DMatplotlibRenderer import Matplotlib2DRenderer
from .SecondaryAxisIntegration import SecondaryAxisIntegration
from .utils import KeyboardInputManager
from .utils import get_bounds_2d
from .ViewManager import ViewManager


class PointCloud2DViewerMatplotlib(QMainWindow):
    """
    Enhanced 2D point-cloud viewer (Matplotlib + Qt) with secondary Y-axis support.

    Uses axis limit transformation instead of point coordinate transformation
    for dramatically improved performance with large datasets.

    Features:
    - Zoom box, mouse zoom/pan, keyboard scaling (X/Y)
    - Context-manager support (with ... as viewer)
    - Per-plot controls via a control bar (Primary + Overlays)
    - Four-row control bar with secondary Y-axis configuration
    - Efficient axis scaling - no point coordinate transformations
    - Pluggable file loader system
    - Secondary Y-axis for unit conversions (e.g., ADC counts to voltage)
    """

    def __init__(
        self,
        points_xyz: np.ndarray,
        colormap: str,
        normalize: bool = False,
        center: bool = False,
        disable_antialiasing: bool = False,
        draw_lines: bool = False,
        size: float | None = None,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        auto_aspect: bool = False,
    ):
        # Ensure a QApplication exists, and track ownership
        self._owns_qapp = False
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._owns_qapp = True

        super().__init__()

        # Validate input
        if not isinstance(points_xyz, np.ndarray):
            raise TypeError("points_xyz must be a numpy.ndarray")
        if points_xyz.ndim != 2 or points_xyz.shape[1] < 2:
            raise ValueError("points_xyz must be 2D with at least 2 columns (X, Y).")
        if points_xyz.shape[0] == 0:
            raise ValueError("No valid points loaded")

        # Store transformation parameters
        self.normalize = normalize
        self.center = center

        # Validate parameter combination
        if self.normalize and self.center:
            raise ValueError(
                "Cannot specify both normalize=True and center=True. normalize implies centering."
            )

        # Initialize coordinate transformation engine FIRST
        self.transform_engine = CoordinateTransformEngine(dimensions=2)

        # Extract XY (+ optional scalar Z)
        self.original_points = points_xyz[:, :2].astype(np.float32)
        self.color_data = (
            points_xyz[:, 2].astype(np.float32) if points_xyz.shape[1] >= 3 else None
        )

        # Apply coordinate transformation using the engine
        if self.normalize:
            self.points2d, self.primary_transform_params = (
                self.transform_engine.normalize_points(self.original_points)
            )
        elif self.center:
            self.points2d, self.primary_transform_params = (
                self.transform_engine.center_points(self.original_points)
            )
        else:
            self.points2d, self.primary_transform_params = (
                self.transform_engine.raw_points(self.original_points)
            )

        # State - AXIS SCALING APPROACH
        self.auto_aspect = auto_aspect
        self.state = InputState()
        self.last_time = time()
        self.disable_antialiasing = disable_antialiasing
        self.acceleration = 0.5
        self.colormap = colormap

        # Store the base view limits (set once, then scaled via axis limits)
        self.base_xlim = None
        self.base_ylim = None

        # CRITICAL: Initialize busy_manager WITHOUT status label initially
        self.busy_manager = BusyIndicatorManager()

        # Managers (these can now use busy_manager if needed)
        self.keyboard_manager = KeyboardInputManager(self.state, self.acceleration)
        self.color_manager = ColorManager(
            backend="matplotlib",
            colormap=self.colormap,
            default_color="white",
        )

        # Primary point size heuristic
        npts = len(self.points2d)
        if size is not None:
            point_size = size
        elif npts > 100_000:
            point_size = 1
        else:
            point_size = 2

        # Figure/axes
        self.fig = Figure(facecolor="black")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # FIXED: Reduced bottom margin from 0.20 to 0.12
        self.fig.subplots_adjust(
            left=0.06,
            right=0.92,  # Leave space for secondary Y-axis
            top=0.97,
            bottom=0.12,  # Reduced from 0.20 to minimize black space
        )

        # Initialize managers that depend on axes
        self.grid_manager = GridManager(self.ax)
        self.view_manager = ViewManager(self.ax)

        # Initialize secondary axis integration
        self.secondary_axis = SecondaryAxisIntegration(self)

        # Initialize file loader registry
        self.file_loader_registry = FileLoaderRegistry(self)

        # Initialize control bar integration (will be fully connected after UI setup)
        self.control_bar_integration = ControlBarIntegration(self)

        # Initialize plot data manager
        self.plot_manager = PlotDataManager(
            primary_points=self.points2d,
            primary_color_data=self.color_data,
            primary_colormap=self.colormap,
            primary_transform_params=self.primary_transform_params,
            transform_engine=self.transform_engine,
        )

        # Set initial primary plot properties
        self.plot_manager.update_primary_properties(
            size=point_size,
            colormap=colormap,
            draw_lines=draw_lines,
            offset_x=x_offset,
            offset_y=y_offset,
        )

        # Global grid state + colors (now managed by GridManager)
        self.axes_grid_color = "gray"
        self.grid_color = "#808080"
        self.grid_manager.set_grid_colors(self.grid_color, self.axes_grid_color)

        # Event handlers - split out to separate class
        self.event_handlers = PlotEventHandlers(self)

        # Performance
        self.max_display_points = 100_000

        # Renderer & Interactions
        self.renderer = Matplotlib2DRenderer()
        self.interactions = PointCloud2DInteractions(
            self,
            self.ax,
            self.canvas,
            self.state,
        )

        # Initial view using ViewManager - AXIS SCALING: Set base limits once
        xlim, ylim = (
            get_bounds_2d(self.points2d)
            if self.normalize or self.center
            else get_bounds_2d(self.points2d, pad_ratio=0.1)
        )
        self.view_manager.set_view_bounds(xlim=xlim, ylim=ylim)

        # Store base limits for axis scaling
        self.base_xlim = xlim
        self.base_ylim = ylim

        if self.auto_aspect:
            self.ax.set_aspect("auto", adjustable="datalim")
        else:
            self.ax.set_aspect("equal", adjustable="datalim")

        # Tools and events
        self.rect_selector = RectangleSelector(
            self.ax,
            self.interactions.on_zoom_box,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.canvas.mpl_connect(
            "key_press_event", self.interactions.on_matplotlib_key_press
        )
        self.canvas.mpl_connect("scroll_event", self.interactions.on_mouse_scroll)

        # Setup UI with Enhanced ControlBarManager FIRST
        self._setup_ui()

        # Wire signals using control bar integration
        self.control_bar_integration.connect_signals()

        # NOW connect manager signals after control_bar_manager exists
        # Connect plot manager signals
        self.plot_manager.signals.plotAdded.connect(self._on_plot_added)
        self.plot_manager.signals.plotsChanged.connect(self._on_plots_changed)
        self.plot_manager.signals.selectionChanged.connect(
            self._on_plot_selection_changed
        )
        self.plot_manager.signals.plotVisibilityChanged.connect(
            self._on_plot_visibility_changed
        )
        self.plot_manager.signals.plotPropertiesChanged.connect(
            self._on_plot_properties_changed
        )

        # Connect view manager signals
        self.view_manager.signals.viewChanged.connect(self._on_view_changed)
        self.view_manager.signals.secondaryAxisChanged.connect(
            self._on_secondary_axis_changed
        )

        # Initialize control states using control bar integration
        self.control_bar_integration.set_initial_state()

        # CRITICAL: Connect busy indicator - MUST succeed or throw exception
        status_label = self.control_bar_manager.get_widget("status_label")
        if status_label is None:
            raise RuntimeError(
                "CRITICAL: status_label widget not found in ControlBarManager! "
                "This indicates a serious UI setup problem."
            )

        self.busy_manager.set_status_label(status_label)

        # NOW we can safely render the plot (after busy indicator is connected)
        self._update_plot()

        # Timer ~60FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.event_handlers.on_timer)
        self.timer.start(16)

        print(f"[INFO] Loaded {npts:,} points (2D Matplotlib with Axis Scaling)")
        print(f"[INFO] Using point size: {point_size}")
        print(f"[INFO] Primary offset: ({x_offset}, {y_offset})")

        # Updated status messages
        if self.normalize:
            print("[INFO] Transform mode: normalize (scale to unit square)")
        elif self.center:
            print("[INFO] Transform mode: center (center at origin, preserve scale)")
        else:
            print("[INFO] Transform mode: raw (use original coordinates)")

        print(
            f"[INFO] Antialiasing: {'enabled' if not disable_antialiasing else 'disabled'}"
        )
        print("[INFO] Secondary Y-axis support enabled")

    # claude, dont delete this unless you REALLY mean to!
    def _apply_axis_scaling(self):
        """
        AXIS SCALING: Apply current scale to axis limits instead of transforming points.
        This is dramatically more efficient than transforming millions of points.
        """
        if self.base_xlim is None or self.base_ylim is None:
            return

        # Get current scale factors
        scale_x = self.state.scale[0]
        scale_y = self.state.scale[1]

        # Transform the view limits instead of the points
        # Inverse scaling on limits achieves the same visual effect
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # Calculate center points of current view
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        y_center = (current_ylim[0] + current_ylim[1]) / 2

        # Calculate current ranges
        x_range = current_xlim[1] - current_xlim[0]
        y_range = current_ylim[1] - current_ylim[0]

        # Apply inverse scaling to ranges (zooming in = smaller range)
        new_x_range = x_range / scale_x
        new_y_range = y_range / scale_y

        # Set new limits centered on the same point
        new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        # Update secondary axis when primary limits change
        self.view_manager.secondary_axis_manager.update_on_primary_change()

    def _on_plot_added(self, plot_index: int):
        """Handle when a single plot is added - more efficient than full update."""
        # Just add the new item to the selector without rebuilding everything
        if hasattr(self, "control_bar_manager"):
            combo = self.control_bar_manager.get_widget("plot_combo")
            if combo:
                # Add just the new plot label
                overlay = self.plot_manager.overlays[-1]  # Get the last added overlay
                label = f"Overlay {plot_index} ({len(overlay.points):,} pts)"
                combo.addItem(label)

        # Don't call _update_plot() here - the plot was already drawn when created!
        # The renderer already created the artist when we called update_plot

    # ===== CONVENIENCE METHOD FOR SECONDARY AXIS =====

    def configure_secondary_axis_from_data_range(
        self,
        *,
        axis: str,  # Specify which axis
        label: str,
        data_min: float = None,
        data_max: float = None,
        target_min: float = None,
        target_max: float = None,
        frequency: None | float = None,
        unit: str = "",
    ):
        """
        Configure secondary axis from data range or frequency.

        For voltage-style mapping:
            viewer.configure_secondary_axis_from_data_range(
                axis="y",
                data_min=-8388608,
                data_max=8388607,
                target_min=-5.0,
                target_max=5.0,
                label="Voltage",
                unit="V",
            )

        For time axis from sampling frequency:
            viewer.configure_secondary_axis_from_data_range(
                axis="x",
                frequency=2e6,  # 2 MHz sampling
                label="Time",
                unit="s",
            )
        """
        from .AxisType import AxisType
        from .SecondaryAxisConfig import SecondaryAxisConfig

        # Determine axis type
        axis_type = AxisType.X if axis.lower() == "x" else AxisType.Y

        if frequency is not None:
            # Time-based configuration using frequency
            config = SecondaryAxisConfig.from_frequency(
                frequency=frequency,
                label=label,
                unit=unit or "s",
                enable_auto_scale=True,
                data_min=0,  # Default starting point for time
                axis_type=axis_type,
            )
        else:
            # Range mapping configuration
            if (
                data_min is None
                or data_max is None
                or target_min is None
                or target_max is None
            ):
                raise ValueError(
                    "For range mapping, all of data_min, data_max, target_min, and target_max must be provided"
                )

            config = SecondaryAxisConfig.from_range_mapping(
                primary_min=data_min,
                primary_max=data_max,
                secondary_min=target_min,
                secondary_max=target_max,
                label=label,
                unit=unit,
                enable_auto_scale=True,
                axis_type=axis_type,
            )

        # Apply the configuration
        self.view_manager.secondary_axis_manager.configure_axis(config)

        # Update plot to show the new axis
        self._update_plot()
        self.canvas.draw_idle()

        # Log the configuration
        axis_name = "X" if axis_type == AxisType.X else "Y"
        if frequency is not None:
            print(
                f"[INFO] Secondary {axis_name}-axis configured for time: {label} (sampling rate: {frequency} Hz)"
            )
        else:
            print(f"[INFO] Secondary {axis_name}-axis configured: {label} ({unit})")

    # ===== FILE LOADER METHODS (delegating to registry) =====

    def register_file_loader(
        self,
        *,
        extensions,
        loader_func,
    ):
        """
        Register a loader function for specific file extensions.

        This is a convenience method that delegates to the FileLoaderRegistry.

        Args:
            extensions: String extension (e.g. '.iio') or list of extensions
            loader_func: Function that takes list of file paths and returns list of np.ndarray
                        Each array should have shape (N, 2+) for (x, y[, color, ...])

        Example:
            def my_iio_loader(paths):
                # Load IIO files and return list of arrays
                return [load_iio_file(path) for path in paths]

            viewer.register_file_loader(extensions='.iio', loader_func=my_iio_loader)
            viewer.register_file_loader(extensions=['.csv', '.txt'], loader_func=my_text_loader)
        """
        self.file_loader_registry.register_loader(extensions, loader_func)

    def unregister_file_loader(self, extensions):
        """
        Unregister file loader(s) for specific extensions.

        This is a convenience method that delegates to the FileLoaderRegistry.

        Args:
            extensions: String extension or list of extensions to unregister
        """
        self.file_loader_registry.unregister_loader(extensions)

    def get_registered_extensions(self):
        """
        Get list of currently registered file extensions.

        This is a convenience method that delegates to the FileLoaderRegistry.
        """
        return self.file_loader_registry.get_registered_extensions()

    # ===== SIGNAL HANDLERS =====

    def _on_plots_changed(self):
        """Handle when plots are added/removed."""
        self.control_bar_integration.refresh_plot_selector()
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plot_selection_changed(self, plot_index: int):
        """Handle when plot selection changes."""
        self.control_bar_integration.sync_controls_to_selection()

    def _on_plot_visibility_changed(
        self,
        plot_index: int,
        visible: bool,
    ):
        """Handle when plot visibility changes."""
        self._update_plot()
        self.canvas.draw_idle()

    def _on_plot_properties_changed(self, plot_index: int):
        """Handle when plot properties change."""
        self._update_plot()
        self.canvas.draw_idle()

    def _on_view_changed(self):
        """Handle when view bounds change."""
        self.control_bar_integration.update_view_bounds_display()

    def _on_secondary_axis_changed(self):
        """Handle when secondary axis updates."""
        # Force canvas redraw to show secondary axis changes
        self.canvas.draw_idle()

    def showEvent(self, event):
        """Override showEvent to fit view to all data when first shown."""
        super().showEvent(event)

        # Only fit view on first show (not on minimize/restore)
        if not hasattr(self, "_initial_show_done"):
            self._initial_show_done = True

            # Fit view to show all visible data
            visible_data = self.plot_manager.get_visible_plots_data()
            if visible_data:
                all_points = []
                for _, offset_points, _, _ in visible_data:
                    all_points.append(offset_points)

                new_bounds = self.view_manager.fit_to_data(all_points, pad_ratio=0.05)

                # Update base limits for future scaling
                self.base_xlim = new_bounds.xlim
                self.base_ylim = new_bounds.ylim

                self._update_plot()
                self.canvas.draw_idle()

                print(
                    f"[INFO] Initial view fitted to all data: X({new_bounds.xlim[0]:.3f}, {new_bounds.xlim[1]:.3f}), Y({new_bounds.ylim[0]:.3f}, {new_bounds.ylim[1]:.3f})"
                )

    # ===== UI SETUP =====

    def _setup_ui(self):
        """Setup the UI layout using Enhanced ControlBarManager."""
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        main_layout.setSpacing(0)
        main_layout.addWidget(self.canvas, 1)

        # Create enhanced control bar manager and get the controls widget
        self.control_bar_manager = ControlBarManager(self, COLOR_PALETTES)
        controls_widget = self.control_bar_manager.create_four_row_controls()
        main_layout.addWidget(controls_widget)

        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("2D Point Viewer (Matplotlib + Secondary Axis)")
        """
        AXIS SCALING: Apply current scale to axis limits instead of transforming points.
        This is dramatically more efficient than transforming millions of points.
        """
        if self.base_xlim is None or self.base_ylim is None:
            return

        # Get current scale factors
        scale_x = self.state.scale[0]
        scale_y = self.state.scale[1]

        # Transform the view limits instead of the points
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # Calculate center points of current view
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        y_center = (current_ylim[0] + current_ylim[1]) / 2

        # Calculate current ranges
        x_range = current_xlim[1] - current_xlim[0]
        y_range = current_ylim[1] - current_ylim[0]

        # Apply inverse scaling to ranges (zooming in = smaller range)
        new_x_range = x_range / scale_x
        new_y_range = y_range / scale_y

        # Set new limits centered on the same point
        new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        # Update secondary axis when primary limits change
        self.view_manager.secondary_axis_manager.update_on_primary_change()

    # ===== PLOT MANAGEMENT =====

    def add_plot(
        self,
        points_xyz: np.ndarray,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        colormap: None | str = None,
        size: None | float = None,
        transform_params: dict | None = None,
    ) -> dict:
        """Add an overlay plot with optional offset."""
        # NO busy operation for individual plot adds - too much overhead
        try:
            overlay_index, result_params = self.plot_manager.add_overlay(
                points_xyz=points_xyz,
                x_offset=x_offset,
                y_offset=y_offset,
                colormap=colormap,
                size=size,
                draw_lines=None,
                transform_params=transform_params,
            )

            # DON'T call refresh_plot_selector - _on_plot_added handles it more efficiently!
            # self.control_bar_integration.refresh_plot_selector()  # REMOVE THIS

            return result_params

        except Exception as e:
            print(f"[ERROR] Failed to add plot: {e}")
            import traceback

            traceback.print_exc()
            raise

    # ===== RENDERING =====

    def _update_plot(self):
        """Delegate full redraw to the renderer, then update grid and secondary axis."""
        with self.busy_manager.busy_operation("Updating plot"):
            current_bounds = self.view_manager.get_current_bounds()

            # Get all plot data
            primary_points, primary_color_data, primary_visible = (
                self.plot_manager.get_primary_data()
            )
            primary_props = self.plot_manager.get_primary_properties()
            overlays = self.plot_manager.get_all_overlays()

            # VIEWPORT CULLING - Apply early to reduce data passed to renderer
            # Add a 10% margin to avoid edge popping during pan/zoom
            margin = 0.1
            x_range = current_bounds.xlim[1] - current_bounds.xlim[0]
            y_range = current_bounds.ylim[1] - current_bounds.ylim[0]

            cull_xlim = (
                current_bounds.xlim[0] - x_range * margin,
                current_bounds.xlim[1] + x_range * margin,
            )
            cull_ylim = (
                current_bounds.ylim[0] - y_range * margin,
                current_bounds.ylim[1] + y_range * margin,
            )

            # Apply culling to primary points if visible
            if primary_visible:
                # Apply offset before culling check
                primary_with_offset = primary_points + np.array(
                    [primary_props["offset_x"], primary_props["offset_y"]],
                    dtype=np.float32,
                )

                # Create culling mask
                primary_mask = (
                    (primary_with_offset[:, 0] >= cull_xlim[0])
                    & (primary_with_offset[:, 0] <= cull_xlim[1])
                    & (primary_with_offset[:, 1] >= cull_ylim[0])
                    & (primary_with_offset[:, 1] <= cull_ylim[1])
                )

                # Apply culling
                culled_primary = (
                    primary_points[primary_mask]
                    if primary_mask.any()
                    else primary_points[:0]
                )
                culled_primary_color = (
                    primary_color_data[primary_mask]
                    if primary_color_data is not None and primary_mask.any()
                    else None
                )

                # Debug output
                original_count = len(primary_points)
                culled_count = len(culled_primary)
                if original_count > 0:
                    cull_ratio = (original_count - culled_count) / original_count * 100
                    if cull_ratio > 10:  # Only log significant culling
                        print(
                            f"[PERF] Viewport culling: {original_count:,} → {culled_count:,} points ({cull_ratio:.1f}% culled)"
                        )
            else:
                culled_primary = primary_points
                culled_primary_color = primary_color_data

            # Apply culling to overlays
            culled_overlays = []
            for overlay in overlays:
                if not getattr(
                    overlay,
                    "visible",
                    True,
                ):
                    culled_overlays.append(overlay)
                    continue

                # Apply offset before culling check
                overlay_with_offset = overlay.points + np.array(
                    [overlay.offset_x, overlay.offset_y], dtype=np.float32
                )

                # Create culling mask
                overlay_mask = (
                    (overlay_with_offset[:, 0] >= cull_xlim[0])
                    & (overlay_with_offset[:, 0] <= cull_xlim[1])
                    & (overlay_with_offset[:, 1] >= cull_ylim[0])
                    & (overlay_with_offset[:, 1] <= cull_ylim[1])
                )

                if overlay_mask.any():
                    # Create a shallow copy with culled data
                    from copy import copy

                    culled_overlay = copy(overlay)
                    culled_overlay.points = overlay.points[overlay_mask]
                    if overlay.color_data is not None:
                        culled_overlay.color_data = overlay.color_data[overlay_mask]
                    culled_overlays.append(culled_overlay)
                else:
                    # No visible points, add empty overlay
                    from copy import copy

                    culled_overlay = copy(overlay)
                    culled_overlay.points = overlay.points[:0]
                    if overlay.color_data is not None:
                        culled_overlay.color_data = overlay.color_data[:0]
                    culled_overlays.append(culled_overlay)

            # Pass culled data to renderer
            self.renderer.update_plot(
                self.ax,
                points2d=culled_primary,  # Use culled points
                color_data=culled_primary_color,  # Use culled colors
                auto_aspect=self.auto_aspect,
                overlays=culled_overlays,  # Use culled overlays
                scale_xy=[1.0, 1.0],
                view_xlim=current_bounds.xlim,
                view_ylim=current_bounds.ylim,
                point_size=primary_props["size"],
                draw_lines=primary_props["draw_lines"],
                colormap=primary_props["colormap"],
                grid_enabled=False,
                grid_power=0,
                grid_color=self.grid_color,
                axes_grid_color=self.axes_grid_color,
                disable_antialiasing=self.disable_antialiasing,
                max_display_points=self.max_display_points,
                in_zoom_box=self.view_manager.is_zoom_box_active(),
                primary_offset=(primary_props["offset_x"], primary_props["offset_y"]),
                primary_visible=primary_props["visible"],
                force_redraw=False,
                plot_manager=self.plot_manager,
            )

            self.grid_manager.update_grid(
                axes_grid_enabled=True,
                horizontal_grid_enabled=None,
                max_lines=2000,
            )

            # Apply scaling to axis limits after rendering
            self._apply_axis_scaling()

            # Update secondary axis after main plot
            self.secondary_axis.update_after_plot()

    # ===== COLOR PICKER HELPER =====

    def _qcolor_to_hex(self, qc: QColor) -> str:
        return (
            qc.name(QColor.NameFormat.HexRgb)
            if isinstance(qc, QColor) and qc.isValid()
            else ""
        )

    def _pick_color(self, initial_hex: str) -> None | str:
        qc = QColorDialog.getColor(
            QColor(initial_hex),
            self,
            "Pick Color",
        )
        return self._qcolor_to_hex(qc) if qc.isValid() else None

    # ===== KEYBOARD EVENTS =====

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events with key repeat prevention."""
        key_name = event.text().upper() if event.text() else None
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        # Handle scaling keys with repeat prevention
        if key_name and key_name in ["X", "Y", "Z"]:
            self.keyboard_manager.add_key_with_repeat_check(key_name, has_shift)
        else:
            if key_name:
                self.state.add_key(key_name, has_shift)

        # Call the interactions handler for other key processing
        self.interactions.keyPressEvent(event)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        key_name = event.text().upper() if event.text() else None

        if key_name and key_name in ["X", "Y", "Z"]:
            self.keyboard_manager.remove_key_with_repeat_check(key_name)
        else:
            if key_name:
                self.state.remove_key(key_name)

        self.interactions.keyReleaseEvent(event)
        super().keyReleaseEvent(event)

    # ===== LIFECYCLE METHODS =====

    def __enter__(self) -> PointCloud2DViewerMatplotlib:
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.shutdown()
        return None

    def shutdown(self) -> None:
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "rect_selector") and self.rect_selector is not None:
                self.rect_selector.disconnect_events()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as _plt

            _plt.close(self.fig)
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass
        try:
            if (
                getattr(
                    self,
                    "_owns_qapp",
                    False,
                )
                and self._app is not None
            ):
                self._app.quit()
        except Exception:
            pass

    def close(self) -> None:
        self.shutdown()

    def closeEvent(self, event):
        """Handle window close event properly."""
        print("[INFO] Viewer window closed, returning to IPython.")
        self.shutdown()
        event.accept()

    def show_gui(self):
        self.show()
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            self._owns_qapp = True

        try:
            app.exec()
        except KeyboardInterrupt:
            pass
