#!/usr/bin/env python3

"""
Enhanced Secondary Axis Configuration with pint unit handling

This module provides configuration for secondary Y-axis with automatic
unit scaling using the pint library for proper unit conversion.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

from .SecondaryAxisConfig import SecondaryAxisConfig

# Integration into ControlBarManager
class EnhancedControlBarManager:
    """Extended control bar with secondary axis controls and auto-scaling support."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # ... existing initialization ...
        self.secondary_axis_widgets = {}

    def _create_secondary_axis_controls(self):
        """Create controls for secondary Y-axis configuration with auto-scaling option."""
        from PyQt6.QtWidgets import QCheckBox
        from PyQt6.QtWidgets import QFormLayout
        from PyQt6.QtWidgets import QGroupBox
        from PyQt6.QtWidgets import QLineEdit
        from PyQt6.QtWidgets import QPushButton

        # Group box for secondary axis controls
        group = QGroupBox("Secondary Y-Axis")
        layout = QFormLayout(group)

        # Enable checkbox
        enable_chk = QCheckBox("Enable")
        enable_chk.toggled.connect(self._on_secondary_axis_toggled)
        layout.addRow(enable_chk)
        self.secondary_axis_widgets["enable"] = enable_chk

        # Auto-scaling checkbox
        auto_scale_chk = QCheckBox("Auto-scale units")
        auto_scale_chk.setChecked(True)  # Default to enabled
        auto_scale_chk.setToolTip(
            "Automatically switch between V, mV, µV, etc. when zooming"
        )
        layout.addRow(auto_scale_chk)
        self.secondary_axis_widgets["auto_scale"] = auto_scale_chk

        # Range mapping inputs
        primary_min_edit = QLineEdit()
        primary_min_edit.setPlaceholderText("e.g., -8388608")
        layout.addRow("Primary Min:", primary_min_edit)
        self.secondary_axis_widgets["primary_min"] = primary_min_edit

        primary_max_edit = QLineEdit()
        primary_max_edit.setPlaceholderText("e.g., 8388607")
        layout.addRow("Primary Max:", primary_max_edit)
        self.secondary_axis_widgets["primary_max"] = primary_max_edit

        secondary_min_edit = QLineEdit()
        secondary_min_edit.setPlaceholderText("e.g., -5.0")
        layout.addRow("Secondary Min:", secondary_min_edit)
        self.secondary_axis_widgets["secondary_min"] = secondary_min_edit

        secondary_max_edit = QLineEdit()
        secondary_max_edit.setPlaceholderText("e.g., 5.0")
        layout.addRow("Secondary Max:", secondary_max_edit)
        self.secondary_axis_widgets["secondary_max"] = secondary_max_edit

        label_edit = QLineEdit()
        label_edit.setPlaceholderText("e.g., Voltage")
        layout.addRow("Label:", label_edit)
        self.secondary_axis_widgets["label"] = label_edit

        unit_edit = QLineEdit()
        unit_edit.setPlaceholderText("e.g., V")
        layout.addRow("Unit:", unit_edit)
        self.secondary_axis_widgets["unit"] = unit_edit

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply_secondary_axis)
        layout.addRow(apply_btn)
        self.secondary_axis_widgets["apply"] = apply_btn

        return group

    def _on_secondary_axis_toggled(self, enabled: bool):
        """Handle secondary axis enable/disable."""
        # Enable/disable input widgets
        for key, widget in self.secondary_axis_widgets.items():
            if key != "enable":
                widget.setEnabled(enabled)

        # Signal to viewer
        if hasattr(self, "secondaryAxisToggled"):
            self.secondaryAxisToggled.emit(enabled)

    def _on_apply_secondary_axis(self):
        """Apply secondary axis configuration with auto-scaling support."""
        try:
            # Get auto-scale setting
            auto_scale = self.secondary_axis_widgets["auto_scale"].isChecked()

            config = SecondaryAxisConfig.from_range_mapping(
                primary_min=float(self.secondary_axis_widgets["primary_min"].text()),
                primary_max=float(self.secondary_axis_widgets["primary_max"].text()),
                secondary_min=float(
                    self.secondary_axis_widgets["secondary_min"].text()
                ),
                secondary_max=float(
                    self.secondary_axis_widgets["secondary_max"].text()
                ),
                label=self.secondary_axis_widgets["label"].text(),
                unit=self.secondary_axis_widgets["unit"].text(),
                enable_auto_scale=auto_scale,
            )

            # Signal to viewer with config
            if hasattr(self, "secondaryAxisConfigChanged"):
                self.secondaryAxisConfigChanged.emit(config)

        except ValueError as e:
            print(f"[ERROR] Invalid secondary axis configuration: {e}")
