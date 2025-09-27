#!/usr/bin/env python3
from __future__ import annotations

import os  # For os._exit and os.devnull
import signal
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from .utils import pad_to_dimensions
from .utils import validate_array
from .utils import validate_colors
from .utils import validate_scalars

# pylint: disable=no-member


class PointCloud2DViewerPolyscope:
    """
    Minimal 2D point viewer using Polyscope with rubber band zoom functionality.
    - Accepts Nx2 points (padded to Z=0).
    - Optional per-point RGB colors in [0,1].
    - Optional scalar quantity for overlays.
    - Camera defaults: Planar (orthographic) with Y-up.
    - Rubber band zoom: Hold 'R' and drag to select a region, release to zoom
    - Quit on 'q' or Escape.
    """

    def __init__(
        self,
        title: str = "mptuple3d: 2D (polyscope)",
        cloud_name: str = "points2d",
        scale_xy: np.ndarray | None = None,
        enable_floor: bool = False,
    ) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be str")
        if not isinstance(cloud_name, str):
            raise TypeError("cloud_name must be str")
        self.title = title
        self.cloud_name = cloud_name
        self._points2d: None | np.ndarray = None
        self._colors: None | np.ndarray = None
        self._scalars: None | tuple[str, np.ndarray] = None
        self._scale_xy = (
            np.asarray(scale_xy, dtype=np.float32).reshape(2)
            if scale_xy is not None
            else np.array([1.0, 1.0], dtype=np.float32)
        )

        # Rubber band zoom state
        self._rubber_band_active = False
        self._rubber_band_start = None
        self._rubber_band_end = None
        self._mouse_down = False
        self._r_key_held = False

        ps.set_program_name(self.title)
        ps.set_verbosity(0)
        # This disables Polyscope's own prefs file (polyscope.ini), not ImGui's:
        ps.set_use_prefs_file(False)

        ps.init()

        # --- ImGui ini suppression: route to /dev/null and clear any loaded state ---
        # Important: use bytes (char*) not str; never read IniFilename back.
        try:
            psim.LoadIniSettingsFromMemory(b"")  # clear any preloaded ini from memory
        except Exception:
            pass
        try:
            psim.GetIO().IniFilename = os.fsencode(os.devnull)  # e.g. b"/dev/null"
        except Exception:
            pass
        # ---------------------------------------------------------------------------

        ps.set_up_dir("y_up")  # Y is up
        ps.set_view_projection_mode("orthographic")  # Planar projection
        ps.set_navigation_style("planar")  # Planar camera controls
        ps.set_background_color((0.0, 0.0, 0.0))
        if enable_floor:
            ps.set_ground_plane_mode("tile")
        else:
            ps.set_ground_plane_mode("none")
        # Ctrl-C exit
        signal.signal(signal.SIGINT, self._sigint_exit)
        # Add key callback via ImGui hook
        ps.set_user_callback(self._on_frame)

    # ---------- public API ----------
    @property
    def scale(self) -> np.ndarray:
        return self._scale_xy

    @scale.setter
    def scale(self, v: np.ndarray) -> None:
        vv = np.asarray(v, dtype=np.float32)
        if vv.shape != (2,):
            raise ValueError("scale must have shape (2,)")
        self._scale_xy = vv
        if self._points2d is not None:
            self._register()

    def set_points(self, points_xy: np.ndarray) -> None:
        # Use consolidated validation
        pts2 = (
            validate_array(
                points_xy,
                "(N, 2)",
                "points",
            )
            * self._scale_xy[None, :]
        )
        self._points2d = pts2
        self._register()

    def set_colors(self, rgb01: np.ndarray) -> None:
        if self._points2d is None:
            raise RuntimeError("set_points() must be called before set_colors()")
        # Use consolidated validation
        self._colors = validate_colors(rgb01, self._points2d.shape[0])
        self._register()

    def set_scalars(
        self,
        label: str,
        values: np.ndarray,
    ) -> None:
        if not isinstance(label, str):
            raise TypeError("label must be str")
        if self._points2d is None:
            raise RuntimeError("set_points() must be called before set_scalars()")
        # Use consolidated validation
        vals = validate_scalars(
            values,
            self._points2d.shape[0],
            "values",
        )
        self._scalars = (label, vals)
        self._register()

    def clear_quantities(self) -> None:
        if self._points2d is None:
            return
        self._colors, self._scalars = None, None
        self._register()

    def show(self) -> None:
        if self._points2d is None:
            raise RuntimeError("No points set. Call set_points() before show().")
        ps.show()

    def close(self) -> None:
        try:
            ps.get_point_cloud(self.cloud_name).remove()
        except Exception:
            try:
                ps.remove_point_cloud(self.cloud_name)
            except Exception:
                try:
                    ps.remove_all_structures()
                except Exception:
                    pass

    # ---------- rubber band zoom functionality ----------

    def fit_to_selection(
        self,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
    ) -> None:
        """Fit camera to view the selected 2D bounding box."""
        if self._points2d is None:
            return

        # Calculate center and size of selection
        center_2d = (min_corner + max_corner) * 0.5
        size_2d = max_corner - min_corner

        # For 2D, we need to set up the camera to look at the XY plane
        # Use the center as the look-at point with Z=0
        center_3d = np.array([center_2d[0], center_2d[1], 0.0])

        # Calculate an appropriate distance based on the selection size
        max_size = max(size_2d[0], size_2d[1])
        distance = max_size * 2.0

        # Position camera above the selection looking down
        camera_pos = center_3d + np.array([0, 0, distance])

        # Set the camera to look at the selection center
        ps.look_at(center_3d, camera_pos)

    def zoom_to_points_in_region(
        self,
        screen_min: tuple[float, float],
        screen_max: tuple[float, float],
    ) -> None:
        """Zoom to the approximate region based on screen selection."""
        if self._points2d is None:
            return

        # Get screen dimensions
        io = psim.GetIO()
        screen_w = io.DisplaySize.x
        screen_h = io.DisplaySize.y

        # Calculate selection center and size
        sel_min_x = max(0, min(screen_min[0], screen_max[0]))
        sel_max_x = min(screen_w, max(screen_min[0], screen_max[0]))
        sel_min_y = max(0, min(screen_min[1], screen_max[1]))
        sel_max_y = min(screen_h, max(screen_min[1], screen_max[1]))

        sel_center_x = (sel_min_x + sel_max_x) * 0.5
        sel_center_y = (sel_min_y + sel_max_y) * 0.5
        sel_width = sel_max_x - sel_min_x
        sel_height = sel_max_y - sel_min_y

        # Get data bounds
        min_pt_2d = self._points2d.min(axis=0)
        max_pt_2d = self._points2d.max(axis=0)
        data_width = max_pt_2d[0] - min_pt_2d[0]
        data_height = max_pt_2d[1] - min_pt_2d[1]
        data_center = (min_pt_2d + max_pt_2d) * 0.5

        # Simple mapping: assume current view shows roughly all data
        # Map screen selection center to approximate data coordinates
        norm_center_x = sel_center_x / screen_w
        norm_center_y = sel_center_y / screen_h

        # Map to data coordinates (with Y flip)
        target_x = min_pt_2d[0] + norm_center_x * data_width
        target_y = max_pt_2d[1] - norm_center_y * data_height  # Flip Y

        # Calculate selection area for zoom level
        selection_area = sel_width * sel_height
        screen_area = screen_w * screen_h
        area_fraction = selection_area / screen_area

        print(
            f"[DEBUG] Screen selection center: ({sel_center_x:.0f}, {sel_center_y:.0f})"
        )
        print(f"[DEBUG] Mapped to data coords: ({target_x:.2f}, {target_y:.2f})")
        print(f"[DEBUG] Selection covers {area_fraction:.1%} of screen")

        try:
            ps.set_view_projection_mode("perspective")

            target = np.array([target_x, target_y, 0.0])
            data_size = max(data_width, data_height)

            # Much more conservative zoom levels
            if area_fraction > 0.6:
                # Large selection - show most of the data
                distance = data_size * 5.0  # Very far back
                print("[DEBUG] Large selection - far view")
            elif area_fraction > 0.3:
                # Medium selection - moderate zoom
                distance = data_size * 4.0  # Still far back
                print("[DEBUG] Medium selection - moderate view")
            elif area_fraction > 0.1:
                # Small selection - closer but not too close
                distance = data_size * 3.0  # Conservative zoom
                print("[DEBUG] Small selection - closer view")
            else:
                # Very small selection - closest zoom but still conservative
                distance = data_size * 2.5  # Still not too close
                print("[DEBUG] Very small selection - closest view")

            camera_pos = target + np.array([0, 0, distance])
            ps.look_at(target, camera_pos)

            print(
                f"[DEBUG] Looking at ({target[0]:.2f}, {target[1]:.2f}) from distance {distance:.2f}"
            )

        except Exception as e:
            print(f"[DEBUG] Zoom failed: {e}")
            # Fallback to data center with safe distance
            try:
                target = np.array([data_center[0], data_center[1], 0.0])
                distance = max(data_width, data_height) * 3.0
                camera_pos = target + np.array([0, 0, distance])
                ps.look_at(target, camera_pos)
                print("[DEBUG] Fallback to data center with safe distance")
            except Exception:
                ps.reset_camera_to_home_view()
                print("[DEBUG] Reset to home view")

        print("[DEBUG] Zoom complete")

    # ---------- internal ----------
    def _sigint_exit(self, *_: object) -> None:
        try:
            self.close()
        finally:
            sys.exit(130)

    def _register(self) -> None:
        assert self._points2d is not None
        # Use consolidated padding function
        pts3 = pad_to_dimensions(self._points2d, 3)
        pc = ps.register_point_cloud(self.cloud_name, pts3)
        if self._colors is not None:
            pc.add_color_quantity(
                "colors",
                self._colors,
                enabled=True,
            )
        if self._scalars is not None:
            label, values = self._scalars
            pc.add_scalar_quantity(
                label,
                values,
                enabled=True,
            )

    def _handle_mouse_input(self) -> None:
        """Handle mouse input for rubber band zoom."""
        io = psim.GetIO()
        mouse_pos = io.MousePos

        # MousePos is a tuple (x, y), not an object with .x, .y attributes
        mouse_x, mouse_y = mouse_pos[0], mouse_pos[1]

        # Check if R key is held (but not shift+R to avoid conflicts)
        shift_held = psim.IsKeyDown(psim.ImGuiKey_LeftShift) or psim.IsKeyDown(
            psim.ImGuiKey_RightShift
        )
        self._r_key_held = psim.IsKeyDown(psim.ImGuiKey_R) and not shift_held

        # Handle mouse down
        if psim.IsMouseClicked(psim.ImGuiMouseButton_Left) and self._r_key_held:
            self._mouse_down = True
            self._rubber_band_start = (mouse_x, mouse_y)
            self._rubber_band_end = self._rubber_band_start
            self._rubber_band_active = True

        # Handle mouse drag
        elif psim.IsMouseDown(psim.ImGuiMouseButton_Left) and self._rubber_band_active:
            self._rubber_band_end = (mouse_x, mouse_y)

        # Handle mouse release
        elif (
            psim.IsMouseReleased(psim.ImGuiMouseButton_Left)
            and self._rubber_band_active
        ):
            if self._rubber_band_start and self._rubber_band_end:
                # Calculate selection bounds
                min_x = min(self._rubber_band_start[0], self._rubber_band_end[0])
                max_x = max(self._rubber_band_start[0], self._rubber_band_end[0])
                min_y = min(self._rubber_band_start[1], self._rubber_band_end[1])
                max_y = max(self._rubber_band_start[1], self._rubber_band_end[1])

                # Only zoom if the selection is big enough
                if abs(max_x - min_x) > 10 and abs(max_y - min_y) > 10:
                    self.zoom_to_points_in_region((min_x, min_y), (max_x, max_y))
                    print(
                        f"[INFO] Zoomed to region: ({min_x:.0f},{min_y:.0f}) to ({max_x:.0f},{max_y:.0f})"
                    )

            # Reset rubber band state
            self._rubber_band_active = False
            self._mouse_down = False
            self._rubber_band_start = None
            self._rubber_band_end = None

    def _draw_mouse_capture_overlay(self) -> None:
        """Draw an invisible overlay to capture mouse when R is held."""
        if not self._r_key_held:
            return

        # Create an invisible button that covers the entire screen
        # This should intercept mouse events before they reach Polyscope
        io = psim.GetIO()

        # Set the next window to cover the entire screen
        psim.SetNextWindowPos((0, 0))
        # Try to get screen dimensions a different way
        try:
            # Try to access DisplaySize components directly
            display_w = io.DisplaySize.x
            display_h = io.DisplaySize.y
        except Exception:
            # Fallback to a large size if we can't get display size
            display_w, display_h = 2000, 2000

        psim.SetNextWindowSize((display_w, display_h))

        # Window flags for an invisible overlay
        window_flags = (
            psim.ImGuiWindowFlags_NoTitleBar
            | psim.ImGuiWindowFlags_NoResize
            | psim.ImGuiWindowFlags_NoMove
            | psim.ImGuiWindowFlags_NoScrollbar
            | psim.ImGuiWindowFlags_NoBackground
            | psim.ImGuiWindowFlags_NoSavedSettings
            | psim.ImGuiWindowFlags_NoFocusOnAppearing
            | psim.ImGuiWindowFlags_NoBringToFrontOnFocus
        )

        expanded, opened = psim.Begin(
            "##RubberBandCapture",
            True,
            window_flags,
        )
        if expanded:
            # Create an invisible button that covers the entire window area
            psim.InvisibleButton("##FullScreenCapture", (display_w, display_h))

            # Also draw a very subtle tint to show selection mode is active
            if psim.IsItemHovered():
                draw_list = psim.GetWindowDrawList()
                window_pos = psim.GetWindowPos()
                overlay_color = psim.GetColorU32(
                    (1.0, 1.0, 0.0, 0.03)
                )  # Very subtle yellow
                draw_list.AddRectFilled(
                    window_pos,
                    (window_pos[0] + display_w, window_pos[1] + display_h),
                    overlay_color,
                )

        psim.End()

    def _draw_rubber_band(self) -> None:
        """Draw the rubber band selection rectangle."""
        if (
            not self._rubber_band_active
            or not self._rubber_band_start
            or not self._rubber_band_end
        ):
            return

        draw_list = psim.GetForegroundDrawList()

        min_x = min(self._rubber_band_start[0], self._rubber_band_end[0])
        max_x = max(self._rubber_band_start[0], self._rubber_band_end[0])
        min_y = min(self._rubber_band_start[1], self._rubber_band_end[1])
        max_y = max(self._rubber_band_start[1], self._rubber_band_end[1])

        # Draw rectangle outline - use tuple format for GetColorU32
        color = psim.GetColorU32((1.0, 1.0, 1.0, 0.8))  # White with alpha
        draw_list.AddRect(
            (min_x, min_y),
            (max_x, max_y),
            color,
            0.0,
            0,
            2.0,
        )

        # Draw semi-transparent fill
        fill_color = psim.GetColorU32((1.0, 1.0, 1.0, 0.1))  # Very transparent white
        draw_list.AddRectFilled(
            (min_x, min_y),
            (max_x, max_y),
            fill_color,
        )

    def _draw_ui_panel(self) -> None:
        """Draw ImGui panel with camera controls and instructions."""
        # Fix for Polyscope's ImGui API - Begin() returns (expanded, opened)
        expanded, opened = psim.Begin("Camera Controls", True)
        if expanded:
            # Instructions
            psim.Text("Rubber Band Zoom (2D):")
            psim.Text("Hold 'R' key and drag to select region")
            psim.Text("Release mouse to zoom to selection")
            psim.Separator()

            # Camera fit buttons
            if psim.Button("Fit All Points"):
                if self._points2d is not None:
                    min_pt_2d = self._points2d.min(axis=0)
                    max_pt_2d = self._points2d.max(axis=0)
                    self.fit_to_selection(min_pt_2d, max_pt_2d)

            psim.SameLine()
            if psim.Button("Reset View"):
                ps.reset_camera_to_home_view()

            # Status
            psim.Separator()
            if self._r_key_held:
                psim.TextColored(
                    (1.0, 1.0, 0.0, 1.0), "R key held - ready for selection"
                )
            if self._rubber_band_active:
                psim.TextColored((0.0, 1.0, 0.0, 1.0), "Selecting region...")

        psim.End()

    def _on_frame(self) -> None:
        """Per-frame callback to catch key presses like 'q'."""
        # Reassert ini suppression every frame WITHOUT reading IniFilename
        try:
            psim.GetIO().IniFilename = os.fsencode(os.devnull)
        except Exception:
            pass

        # Use ImGuiKey enums, not raw ints
        if psim.IsKeyPressed(psim.ImGuiKey_Q, False) or psim.IsKeyPressed(
            psim.ImGuiKey_Escape, False
        ):
            print("[INFO] 'q' pressed, closing Polyscope.")
            # Immediately terminate the process to avoid "mid-frame shutdown" error.
            os._exit(0)

        # Draw mouse capture overlay FIRST (before other UI elements)
        # This ensures it can intercept mouse events before they reach Polyscope
        self._draw_mouse_capture_overlay()

        # Handle mouse input for rubber band zoom
        self._handle_mouse_input()

        # Draw UI elements
        self._draw_ui_panel()
        self._draw_rubber_band()


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 5000
    pts = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    rad = np.sqrt((pts**2).sum(axis=1)).astype(np.float32)
    colors = (
        np.stack([rad, 1.0 - rad, np.zeros_like(rad)], axis=1)
        .clip(0.0, 1.0)
        .astype(np.float32)
    )
    viewer = PointCloud2DViewerPolyscope(
        title="Demo: 2D Polyscope with Rubber Band Zoom", cloud_name="demo2d"
    )
    viewer.set_points(pts)
    viewer.set_colors(colors)
    viewer.set_scalars("radius", rad)
    viewer.show()
