#!/usr/bin/env python3
from __future__ import annotations

# pylint: disable=no-member
import os
import signal
import sys

import numpy as np
import polyscope.imgui as psim

try:
    import polyscope as ps
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import polyscope: {exc}") from exc

from .utils import validate_array
from .utils import validate_colors
from .utils import validate_scalars


class PointCloud3DViewerPolyscope:
    """
    3D point cloud viewer using Polyscope with rubber band zoom functionality.

    API (mirrors VisPy viewers):
      - call `set_points()` (required) and optional `set_colors()`, `set_scalars()`
      - call `show()` to open the interactive window
      - subsequent `set_*()` calls re-register the cloud under the same name

    Features:
      - Rubber band zoom: Hold 'R' and drag to select a region, release to zoom
      - Fit to selection functionality
      - Camera position controls via ImGui
    """

    def __init__(
        self,
        title: str = "mptuple3d: 3D (polyscope)",
        cloud_name: str = "points3d",
        scale: np.ndarray | None = None,
        enable_floor: bool = False,
    ) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be str")
        if not isinstance(cloud_name, str):
            raise TypeError("cloud_name must be str")

        self.title = title
        self.cloud_name = cloud_name
        self._points: None | np.ndarray = None
        self._colors: None | np.ndarray = None
        self._scalars: None | tuple[str, np.ndarray] = None
        self._scale = (
            np.asarray(scale, dtype=np.float32).reshape(3)
            if scale is not None
            else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )

        # Rubber band zoom state
        self._rubber_band_active = False
        self._rubber_band_start = None
        self._rubber_band_end = None
        self._mouse_down = False
        self._r_key_held = False

        # Program options before init
        ps.set_program_name(self.title)
        ps.set_verbosity(0)
        ps.set_use_prefs_file(False)
        ps.init()

        # Scene options
        ps.set_up_dir("y_up")
        ps.set_background_color((0.0, 0.0, 0.0))

        if enable_floor:
            ps.set_ground_plane_mode("tile")
        else:
            ps.set_ground_plane_mode("none")

        # Graceful Ctrl-C
        signal.signal(signal.SIGINT, self._sigint_exit)
        # Set the frame callback
        ps.set_user_callback(self._on_frame)

    # ---------- public API ----------

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    def scale(self, v: np.ndarray) -> None:
        vv = np.asarray(v, dtype=np.float32)
        if vv.shape != (3,):
            raise ValueError("scale must have shape (3,)")
        self._scale = vv
        if self._points is not None:
            self._register()

    def set_points(self, points_xyz: np.ndarray) -> None:
        # Use consolidated validation
        pts = (
            validate_array(
                points_xyz,
                "(N, 3)",
                "points",
            )
            * self._scale[None, :]
        )
        self._points = pts
        self._register()

    def set_colors(self, rgb01: np.ndarray) -> None:
        if self._points is None:
            raise RuntimeError("set_points() must be called before set_colors()")
        # Use consolidated validation
        self._colors = validate_colors(rgb01, self._points.shape[0])
        self._register()

    def set_scalars(
        self,
        label: str,
        values: np.ndarray,
    ) -> None:
        if not isinstance(label, str):
            raise TypeError("label must be str")
        if self._points is None:
            raise RuntimeError("set_points() must be called before set_scalars()")
        # Use consolidated validation
        vals = validate_scalars(
            values,
            self._points.shape[0],
            "values",
        )
        self._scalars = (label, vals)
        self._register()

    def clear_quantities(self) -> None:
        if self._points is None:
            return
        self._colors, self._scalars = None, None
        self._register()

    def show(self) -> None:
        if self._points is None:
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
        """Fit camera to view the selected bounding box."""
        if self._points is None:
            return

        # Calculate center and size of selection
        center = (min_corner + max_corner) * 0.5
        size = np.linalg.norm(max_corner - min_corner)

        # Set camera to look at the center from an appropriate distance
        # The distance calculation is heuristic - adjust as needed
        distance = size * 2.0

        # Use Polyscope's camera positioning
        # Note: Polyscope's camera API is limited, so we use what's available
        ps.look_at(center, center + np.array([0, 0, distance]))

    def zoom_to_points_in_region(
        self,
        screen_min: tuple[float, float],
        screen_max: tuple[float, float],
    ) -> None:
        """Zoom to points that fall within the screen region."""
        if self._points is None:
            return

        # Calculate selection center and size
        sel_center_x = (screen_min[0] + screen_max[0]) * 0.5
        sel_center_y = (screen_min[1] + screen_max[1]) * 0.5
        sel_width = abs(screen_max[0] - screen_min[0])
        sel_height = abs(screen_max[1] - screen_min[1])

        try:
            # Get screen dimensions
            io = psim.GetIO()
            screen_w = io.DisplaySize.x
            screen_h = io.DisplaySize.y
        except:
            screen_w, screen_h = 1024, 768

        # Convert screen selection center to normalized coordinates [-1, 1]
        # Screen coordinates: (0,0) = top-left, (screen_w, screen_h) = bottom-right
        # Normalized: (-1,-1) = bottom-left, (1,1) = top-right
        norm_x = (sel_center_x / screen_w) * 2.0 - 1.0
        norm_y = 1.0 - (sel_center_y / screen_h) * 2.0  # Flip Y axis

        # Estimate a 3D target point based on the screen selection
        # This is a simplified approach since we don't have full camera intrinsics

        # Get point cloud bounds
        min_pt = self._points.min(axis=0)
        max_pt = self._points.max(axis=0)
        cloud_center = (min_pt + max_pt) * 0.5
        cloud_size = np.linalg.norm(max_pt - min_pt)

        # Simple heuristic: map normalized screen coordinates to a plane in front of the point cloud
        # This assumes we're looking roughly along the Z axis
        offset_scale = cloud_size * 0.5  # How much the selection can offset from center
        target_x = cloud_center[0] + norm_x * offset_scale
        target_y = cloud_center[1] + norm_y * offset_scale
        target_z = cloud_center[2]  # Keep same Z as cloud center

        target_point = np.array([target_x, target_y, target_z])

        # Calculate zoom factor based on selection size
        selection_area = sel_width * sel_height
        screen_area = screen_w * screen_h
        area_ratio = selection_area / screen_area
        zoom_factor = max(0.1, min(10.0, 1.0 / max(area_ratio, 0.01)))

        # Set camera distance based on zoom factor
        base_distance = cloud_size * 1.5
        new_distance = base_distance / zoom_factor

        # Look at the estimated target point instead of always the cloud center
        camera_pos = target_point + np.array([0, 0, new_distance])

        print(f"[DEBUG] Selection center: ({sel_center_x:.0f}, {sel_center_y:.0f})")
        print(f"[DEBUG] Normalized: ({norm_x:.2f}, {norm_y:.2f})")
        print(f"[DEBUG] Target point: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        print(f"[DEBUG] Zoom factor: {zoom_factor:.2f}, Distance: {new_distance:.2f}")

        # Apply the zoom to the target point
        ps.look_at(target_point, camera_pos)

    # ---------- internal ----------

    def _sigint_exit(self, *_: object) -> None:
        try:
            self.close()
        finally:
            sys.exit(130)

    def _register(self) -> None:
        assert self._points is not None
        pc = ps.register_point_cloud(self.cloud_name, self._points)

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
        except:
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
            psim.Text("Rubber Band Zoom:")
            psim.Text("Hold 'R' key and drag to select region")
            psim.Text("Release mouse to zoom to selection")
            psim.Separator()

            # Camera fit buttons
            if psim.Button("Fit All Points"):
                if self._points is not None:
                    min_pt = self._points.min(axis=0)
                    max_pt = self._points.max(axis=0)
                    self.fit_to_selection(min_pt, max_pt)

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
        # Handle basic key presses for quitting
        if psim.IsKeyPressed(psim.ImGuiKey_Q, False) or psim.IsKeyPressed(
            psim.ImGuiKey_Escape, False
        ):
            print("[INFO] 'q' pressed, closing Polyscope.")
            os._exit(0)

        # Draw mouse capture overlay FIRST (before other UI elements)
        # This ensures it can intercept mouse events before they reach Polyscope
        self._draw_mouse_capture_overlay()

        # Handle mouse input for rubber band zoom
        self._handle_mouse_input()

        # Draw UI elements
        self._draw_ui_panel()
        self._draw_rubber_band()


# Example usage
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(10_000, 3)).astype(np.float32)
    # normalize to unit-ish sphere
    norms = (
        np.linalg.norm(
            pts,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )
    pts = pts / norms

    # simple colorization
    minv = pts.min(axis=0)
    rngv = np.ptp(pts, axis=0) + 1e-12
    colors = ((pts - minv) / rngv).astype(np.float32)

    viewer = PointCloud3DViewerPolyscope(
        title="Demo: 3D Polyscope with Rubber Band Zoom",
        cloud_name="demo3d",
    )
    viewer.set_points(pts)
    viewer.set_colors(colors)
    viewer.set_scalars("radius_like", np.linalg.norm(pts, axis=1).astype(np.float32))
    viewer.show()
