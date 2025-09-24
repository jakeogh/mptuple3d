#!/usr/bin/env python3

"""
Dual Secondary Axis Manager for managing both X and Y secondary axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Optional

from matplotlib.axes import Axes

from .AxisType import AxisType
from .SecondaryAxisManager import SecondaryAxisManager

if TYPE_CHECKING:
    from .SecondaryAxisConfig import SecondaryAxisConfig


class DualSecondaryAxisManager:
    """Manages both X and Y secondary axes for a matplotlib plot."""

    def __init__(self, primary_ax: Axes):
        """Initialize dual secondary axis manager."""
        self.primary_ax = primary_ax
        self.x_axis_manager = SecondaryAxisManager(primary_ax, AxisType.X)
        self.y_axis_manager = SecondaryAxisManager(primary_ax, AxisType.Y)

    def configure_axis(self, config: SecondaryAxisConfig) -> None:
        """Configure the appropriate secondary axis based on config.axis_type."""
        if config.axis_type == AxisType.Y:
            self.y_axis_manager.enable_secondary_axis(config)
        else:
            self.x_axis_manager.enable_secondary_axis(config)

    def disable_axis(self, axis_type: AxisType) -> None:
        """Disable a specific secondary axis."""
        if axis_type == AxisType.Y:
            self.y_axis_manager.disable_secondary_axis()
        else:
            self.x_axis_manager.disable_secondary_axis()

    def update_on_primary_change(self) -> None:
        """Update both secondary axes when primary axes change."""
        self.x_axis_manager.update_on_primary_change()
        self.y_axis_manager.update_on_primary_change()

    def is_axis_enabled(self, axis_type: AxisType) -> bool:
        """Check if a specific axis is enabled."""
        if axis_type == AxisType.Y:
            return self.y_axis_manager.is_enabled()
        else:
            return self.x_axis_manager.is_enabled()

    def get_axis_config(self, axis_type: AxisType) -> SecondaryAxisConfig | None:
        """Get configuration for a specific axis."""
        if axis_type == AxisType.Y:
            return self.y_axis_manager.config
        else:
            return self.x_axis_manager.config

    def is_any_enabled(self) -> bool:
        """Check if any secondary axis is enabled."""
        return self.x_axis_manager.is_enabled() or self.y_axis_manager.is_enabled()

    def disable_all(self) -> None:
        """Disable all secondary axes."""
        self.x_axis_manager.disable_secondary_axis()
        self.y_axis_manager.disable_secondary_axis()

    def get_enabled_axes(self) -> list[AxisType]:
        """Get list of enabled axis types."""
        enabled = []
        if self.x_axis_manager.is_enabled():
            enabled.append(AxisType.X)
        if self.y_axis_manager.is_enabled():
            enabled.append(AxisType.Y)
        return enabled
