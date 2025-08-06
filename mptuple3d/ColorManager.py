#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def make_colors_from_scalar(
    color_data: np.ndarray,
    colormap: str,
    backend: str,
) -> np.ndarray:
    """
    Map scalar data to colors using specified colormap.

    Args:
        color_data: (N,) array of scalar values
        colormap: Name of colormap to use
        backend: "vispy", "matplotlib", or "datoviz"

    Returns:
        Color array - format depends on backend
    """
    if color_data is None:
        return None

    # Normalize to [0, 1]
    color_norm = (color_data - color_data.min()) / max(np.ptp(color_data), 1e-6)

    if backend == "vispy":
        from vispy.color import \
            Colormap  # type: ignore  # pylint: disable=import-outside-toplevel

        if colormap == "viridis":
            cmap = Colormap(["blue", "cyan", "green", "yellow", "red"])
        else:
            cmap = Colormap(colormap)
        return cmap.map(color_norm)

    elif backend == "matplotlib":
        from matplotlib.colors import \
            Colormap  # pylint: disable=import-outside-toplevel

        if colormap == "viridis":
            # pylint: disable-next=no-member
            cmap = Colormap.from_list(
                "custom", ["blue", "cyan", "green", "yellow", "red"]
            )
        else:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

            cmap = plt.get_cmap(colormap)
        return cmap(color_norm)

    elif backend == "datoviz":
        import datoviz as dvz  # type: ignore # pylint: disable=import-outside-toplevel

        s = color_norm.astype(np.float32)
        return dvz.cmap(colormap, s)

    else:
        raise ValueError(f"Unknown backend: {backend}")


class ColorManager:
    """
    Consolidated color generation for all viewers.
    Handles the common pattern of converting scalar data to colors.
    """

    def __init__(
        self,
        backend: str,
        colormap: str,
        default_color: str = "white",
    ):
        """
        Initialize color manager with backend-specific settings.

        Args:
            backend: "vispy", "matplotlib", or "datoviz"
            colormap: Name of colormap to use
            default_color: Default color when no color data provided
        """
        self.backend = backend
        self.colormap = colormap
        self.default_color = default_color

    def make_colors(self, color_data: np.ndarray | None) -> np.ndarray | str:
        """
        Generate colors from color data or return default - exactly matches existing behavior.

        Args:
            color_data: Optional array of scalar values for coloring

        Returns:
            Color array (backend-specific format) or default color string
        """
        if color_data is not None:
            return make_colors_from_scalar(
                color_data,
                colormap=self.colormap,
                backend=self.backend,
            )
        return self.default_color
