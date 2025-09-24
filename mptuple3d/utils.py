#!/usr/bin/env python3
"""
Consolidated utility functions for point cloud processing and visualization.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import time
from typing import Tuple

import numpy as np
from unmp import unmp

# Import KeyboardInputManager from its dedicated file instead of duplicating it
from .KeyboardInputManager import KeyboardInputManager


def normalize_points(points: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """
    Normalize point cloud to fit in a unit cube/square centered at origin.

    Args:
        points: (N, D) array of points
        dimensions: Number of dimensions to consider (None = all)

    Returns:
        Normalized points array
    """
    if points.shape[0] == 0:
        return points

    if dimensions is not None:
        points = points[:, :dimensions]

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    size = max_vals - min_vals
    max_extent = size.max()
    scale_factors = np.where(
        size != 0,
        max_extent / size,
        1,
    )
    return (points - center) * scale_factors


def center_points(points: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """
    Center point cloud at origin without scaling.

    Args:
        points: (N, D) array of points
        dimensions: Number of dimensions to consider (None = all)

    Returns:
        Centered points array
    """
    if points.shape[0] == 0:
        return points

    if dimensions is not None:
        points = points[:, :dimensions]

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    center = (min_vals + max_vals) / 2
    return points - center


def compute_bounds(
    points: np.ndarray,
    pad_ratio: float = 0.05,
    return_format: str = "tuple",
) -> (
    tuple[tuple[float, float] | tuple[float, float]] | tuple[float, float, float, float]
):
    """
    Calculate bounds with padding for initial view.

    Args:
        points: (N, 2) or (N, 3) array of points
        pad_ratio: Padding ratio (fraction of size)
        return_format: "tuple" for ((xmin, xmax), (ymin, ymax)) or "rect" for (x, y, width, height)

    Returns:
        Bounds in requested format
    """
    if points.shape[0] == 0:
        if return_format == "rect":
            return (0.0, 0.0, 1.0, 1.0)
        return (0.0, 1.0), (0.0, 1.0)

    # Use only first 2 dimensions for bounds calculation
    points_2d = points[:, :2] if points.shape[1] >= 2 else points

    min_vals = points_2d.min(axis=0)
    max_vals = points_2d.max(axis=0)
    size = np.maximum(max_vals - min_vals, 1e-12)
    pad = size * pad_ratio
    lo = min_vals - pad
    hi = max_vals + pad

    if return_format == "rect":
        # Return (x, y, width, height) format
        width, height = (hi - lo).tolist()
        return (
            float(lo[0]),
            float(lo[1]),
            float(max(width, 1e-6)),
            float(max(height, 1e-6)),
        )
    else:
        # Return ((xmin, xmax), (ymin, ymax)) format
        return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def validate_array(
    array: np.ndarray,
    expected_shape: tuple[int, ...] | str,
    name: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Validate and convert array to expected format.

    Args:
        array: Input array
        expected_shape: Expected shape tuple or description like "(N, 3)"
        name: Name for error messages
        dtype: Target dtype

    Returns:
        Validated and converted array
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(array).__name__}")

    # Convert dtype if needed
    if array.dtype != dtype:
        try:
            array = array.astype(dtype, copy=False)
        except Exception as exc:
            raise TypeError(f"{name} must be convertible to {dtype}: {exc}") from exc

    # Validate shape
    if isinstance(expected_shape, str):
        # Parse shape description
        if expected_shape == "(N, 2)":
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
        elif expected_shape == "(N, 3)":
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
        elif expected_shape.startswith("(N,"):
            # Extract number of columns
            cols = int(expected_shape.split(",")[1].strip(" )"))
            if array.ndim != 2 or array.shape[1] != cols:
                raise ValueError(
                    f"{name} must have shape (N, {cols}), got {array.shape}"
                )
        elif expected_shape == "(N,)":
            if array.ndim != 1:
                raise ValueError(f"{name} must have shape (N,), got {array.shape}")
    else:
        # Direct shape comparison
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )

    # Check for invalid values
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf values")

    return array


def validate_colors(colors: np.ndarray, n_points: int) -> np.ndarray:
    """
    Validate RGB color array.

    Args:
        colors: (N, 3) array of RGB values in [0, 1]
        n_points: Expected number of points

    Returns:
        Validated color array
    """
    colors = validate_array(
        colors,
        f"({n_points}, 3)",
        "colors",
    )

    if (colors < 0).any() or (colors > 1).any():
        raise ValueError("colors must be in range [0, 1]")

    return colors


def validate_scalars(
    values: np.ndarray,
    n_points: int,
    name: str,
) -> np.ndarray:
    """
    Validate scalar value array.

    Args:
        values: (N,) array of scalar values
        n_points: Expected number of points
        name: Name for error messages

    Returns:
        Validated scalar array
    """
    return validate_array(
        values,
        f"({n_points},)",
        name,
    )


def load_points_from_stdin_ndarray(
    minimum_dimensions: int,
) -> np.ndarray | None:
    """
    Read points from stdin via messagepack.

    Args:
        minimum_dimensions: 2 for (x,y,[color]),
                            3 for (x,y,z,[color]),
                            4 for (x,y,z,color),

    Returns:
        Tuple of (points_array)
    """
    iterator: Sequence[tuple] = unmp(valid_types=[tuple])
    points = []
    found_data = False

    for index, _mpobject in enumerate(iterator):
        _v = _mpobject
        if isinstance(_v, dict):
            # Accept single k:v dict rows, use the value part
            for _, __v in _v.items():
                _v = __v
                break

        # Skip comment lines
        if isinstance(_v[0], str) and _v[0].startswith("#"):
            continue

        # Skip header lines (non-numeric first element)
        if not found_data:
            try:
                float(_v[0])
            except Exception:
                continue

        # Process data based on expected dimensions
        if minimum_dimensions == 2 and len(_v) >= 2:
            x, y = float(_v[0]), float(_v[1])
            found_data = True
            # Third component is color for 2D
            if len(_v) >= 3:
                z = float(_v[2])
                points.append([x, y, z])
            else:
                points.append([x, y])

        elif minimum_dimensions == 3 and len(_v) >= 3:
            x, y, z = float(_v[0]), float(_v[1]), float(_v[2])
            found_data = True
            # Fourth component is color for 3D
            if len(_v) >= 4:
                c = float(_v[3])
                points.append([x, y, z, c])
            else:
                points.append([x, y, z])
        else:
            raise NotImplementedError(f"{minimum_dimensions=}")

    if not points:
        empty_shape = (0, minimum_dimensions)
        return np.empty(empty_shape, dtype=np.float32), None

    pts = np.asarray(points, dtype=np.float32)
    return pts


def load_points_from_stdin(
    expected_dimensions: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read points from stdin via messagepack.

    Args:
        expected_dimensions: 2 for (x,y[,color]), 3 for (x,y,z[,color])

    Returns:
        Tuple of (points_array, color_data_array_or_None)
    """
    iterator: Sequence[tuple] = unmp(valid_types=[tuple])
    points, colors = [], []
    found_data = False

    for index, _mpobject in enumerate(iterator):
        _v = _mpobject
        if isinstance(_v, dict):
            # Accept single k:v dict rows, use the value part
            for _, __v in _v.items():
                _v = __v
                break

        # Skip comment lines
        if isinstance(_v[0], str) and _v[0].startswith("#"):
            continue

        # Skip header lines (non-numeric first element)
        if not found_data:
            try:
                float(_v[0])
            except Exception:
                continue

        # Process data based on expected dimensions
        if expected_dimensions == 2 and len(_v) >= 2:
            x, y = float(_v[0]), float(_v[1])
            found_data = True
            points.append([x, y])
            # Third component is color for 2D
            if len(_v) >= 3:
                colors.append(float(_v[2]))

        elif expected_dimensions == 3 and len(_v) >= 3:
            x, y, z = float(_v[0]), float(_v[1]), float(_v[2])
            found_data = True
            points.append([x, y, z])
            # Fourth component is color for 3D
            if len(_v) >= 4:
                colors.append(float(_v[3]))

    if not points:
        empty_shape = (0, expected_dimensions)
        return np.empty(empty_shape, dtype=np.float32), None

    pts = np.asarray(points, dtype=np.float32)
    clr = np.asarray(colors, dtype=np.float32) if colors else None
    return pts, clr


def pad_to_dimensions(
    points: np.ndarray,
    target_dims: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad points array to target number of dimensions.

    Args:
        points: (N, D) input points
        target_dims: Target number of dimensions
        pad_value: Value to use for padding

    Returns:
        (N, target_dims) padded array
    """
    if points.shape[1] >= target_dims:
        return points[:, :target_dims]

    n_points = points.shape[0]
    n_pad = target_dims - points.shape[1]
    padding = np.full(
        (n_points, n_pad),
        pad_value,
        dtype=points.dtype,
    )
    return np.concatenate([points, padding], axis=1)


# Convenience functions for specific use cases
def normalize_points_2d(points_xy: np.ndarray) -> np.ndarray:
    """Normalize 2D points to unit square centered at origin."""
    return normalize_points(points_xy, dimensions=2)


def center_points_2d(points_xy: np.ndarray) -> np.ndarray:
    """Center 2D points at origin without scaling."""
    return center_points(points_xy, dimensions=2)


def get_bounds_2d(
    points_xy: np.ndarray, pad_ratio: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculate 2D bounds with padding for initial view."""
    return compute_bounds(
        points_xy,
        pad_ratio,
        return_format="tuple",
    )


def get_rect_from_points(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Compute a reasonable PanZoomCamera rect from points."""
    return compute_bounds(
        points_xy,
        pad_ratio=0.05,
        return_format="rect",
    )


def pad_to_3d(points_xy: np.ndarray) -> np.ndarray:
    """Pad 2D points to 3D with Z=0."""
    return pad_to_dimensions(
        points_xy,
        3,
        pad_value=0.0,
    )


def load_points_from_stdin_for_2d() -> tuple[np.ndarray, np.ndarray | None]:
    """Read 2D points from stdin."""
    return load_points_from_stdin(expected_dimensions=2)


def load_points_from_stdin_for_3d() -> tuple[np.ndarray, np.ndarray | None]:
    """Read 3D points from stdin."""
    return load_points_from_stdin(expected_dimensions=3)
