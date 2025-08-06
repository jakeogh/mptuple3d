#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

# pylint: disable=no-name-in-module
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .css import STYLE


class PointCloud2DControlsBar(QWidget):
    """Two-row control bar with better layout and labeling."""

    # Define all signals
    addRequested = pyqtSignal()
    plotChanged = pyqtSignal(int)
    visibilityToggled = pyqtSignal(bool)  # NEW: Signal for visibility changes
    accelChanged = pyqtSignal(float)
    sizeChanged = pyqtSignal(float)
    linesToggled = pyqtSignal(bool)
    paletteChanged = pyqtSignal(str)
    axesGridColorPickRequested = pyqtSignal()
    adcGridColorPickRequested = pyqtSignal()
    gridSpacingChanged = pyqtSignal(str)
    offsetChanged = pyqtSignal(float, float)
    resetRequested = pyqtSignal()
    exitRequested = pyqtSignal()

    def __init__(
        self,
        palette_groups,
        parent=None,
    ):
        super().__init__(parent)
        self._palette_groups = palette_groups

        # Create the two-row layout from scratch
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )
        main_layout.setSpacing(4)

        # First row: Plot controls
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        # Add… button
        self.add_btn = QPushButton("Add…")
        self.add_btn.setMaximumWidth(60)
        self.add_btn.clicked.connect(self.addRequested.emit)
        row1.addWidget(self.add_btn)

        # Plot selector
        lbl_plot = QLabel("Plot:")
        # lbl_plot.setStyleSheet("color: white;")
        row1.addWidget(lbl_plot)

        self.plot_combo = QComboBox()
        self.plot_combo.setMaximumWidth(220)
        self.plot_combo.currentIndexChanged.connect(self.plotChanged.emit)
        row1.addWidget(self.plot_combo)

        # NEW: Visible checkbox
        self.visible_chk = QCheckBox("Visible")
        self.visible_chk.setChecked(True)  # Default to visible
        self.visible_chk.toggled.connect(self.visibilityToggled.emit)
        row1.addWidget(self.visible_chk)

        # Acceleration
        lbl_accel = QLabel("Accel:")
        # lbl_accel.setStyleSheet("color: white;")
        row1.addWidget(lbl_accel)

        self.accel_spin = QDoubleSpinBox()
        self.accel_spin.setRange(1.001, 5.0)
        self.accel_spin.setSingleStep(0.01)
        self.accel_spin.setDecimals(3)
        self.accel_spin.setMaximumWidth(80)
        self.accel_spin.valueChanged.connect(self.accelChanged.emit)
        row1.addWidget(self.accel_spin)

        # Size
        lbl_size = QLabel("Size:")
        # lbl_size.setStyleSheet("color: white;")
        row1.addWidget(lbl_size)

        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 1000.0)
        self.size_spin.setSingleStep(0.1)
        self.size_spin.setDecimals(3)
        self.size_spin.setMaximumWidth(80)
        self.size_spin.valueChanged.connect(self.sizeChanged.emit)
        row1.addWidget(self.size_spin)

        # Lines
        self.lines_chk = QCheckBox("Lines")
        # self.lines_chk.setStyleSheet("color: white;")
        self.lines_chk.toggled.connect(self.linesToggled.emit)
        row1.addWidget(self.lines_chk)

        # Palette
        lbl_palette = QLabel("Palette:")
        # lbl_palette.setStyleSheet("color: white;")
        row1.addWidget(lbl_palette)

        self.palette_combo = QComboBox()
        self.palette_combo.setMaximumWidth(160)
        self._populate_palettes()
        # FIXED: Connect to BOTH signals for immediate feedback when using arrow keys
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        self.palette_combo.currentIndexChanged.connect(self._on_palette_index_changed)
        row1.addWidget(self.palette_combo)

        row1.addStretch()

        # Second row: Grid and offset controls
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        # Axes Grid
        lbl_axes_grid = QLabel("Axes Grid:")
        row2.addWidget(lbl_axes_grid)

        self.axes_grid_btn = QPushButton("Pick")
        self.axes_grid_btn.clicked.connect(self.axesGridColorPickRequested.emit)
        row2.addWidget(self.axes_grid_btn)

        # ADC Grid
        lbl_adc_grid = QLabel("Grid 2^N:")
        row2.addWidget(lbl_adc_grid)

        self.adc_grid_btn = QPushButton("Pick")
        self.adc_grid_btn.clicked.connect(self.adcGridColorPickRequested.emit)
        row2.addWidget(self.adc_grid_btn)

        # Grid spacing
        lbl_spacing = QLabel("Spacing:")
        row2.addWidget(lbl_spacing)

        self.grid_combo = QComboBox()
        self.grid_combo.setMaximumWidth(110)
        self._populate_grid()
        self.grid_combo.currentTextChanged.connect(self.gridSpacingChanged.emit)
        row2.addWidget(self.grid_combo)

        # Offset X (FIXED LABEL)
        lbl_offset_x = QLabel("X:")
        row2.addWidget(lbl_offset_x)

        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-1e12, 1e12)
        self.offset_x_spin.setDecimals(6)
        self.offset_x_spin.setSingleStep(0.1)
        self.offset_x_spin.setMaximumWidth(100)
        self.offset_x_spin.valueChanged.connect(self._emit_offset_changed)
        row2.addWidget(self.offset_x_spin)

        # Offset Y
        lbl_offset_y = QLabel("Y:")
        row2.addWidget(lbl_offset_y)

        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-1e12, 1e12)
        self.offset_y_spin.setDecimals(6)
        self.offset_y_spin.setSingleStep(0.1)
        self.offset_y_spin.setMaximumWidth(100)
        self.offset_y_spin.valueChanged.connect(self._emit_offset_changed)
        row2.addWidget(self.offset_y_spin)

        # Reset / Exit
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setMaximumWidth(90)
        self.reset_btn.clicked.connect(self.resetRequested.emit)
        row2.addWidget(self.reset_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setMaximumWidth(60)
        self.exit_btn.clicked.connect(self.exitRequested.emit)
        row2.addWidget(self.exit_btn)

        # Info label
        self.info_label = QLabel("")
        row2.addWidget(self.info_label)

        row2.addStretch()

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)

        self.setStyleSheet(STYLE)

    def _populate_palettes(self):
        self.palette_combo.clear()
        for category, palettes in self._palette_groups.items():
            self.palette_combo.addItem(f"───『{category}』───")
            idx = self.palette_combo.count() - 1
            item = self.palette_combo.model().item(idx)
            item.setEnabled(False)
            for p in palettes:
                self.palette_combo.addItem(p)

    def _populate_grid(self):
        self.grid_combo.clear()
        self.grid_combo.addItem("Off")
        for n in range(1, 25):
            spacing = 2**n
            self.grid_combo.addItem(f"2^{n} ({spacing})" if n <= 10 else f"2^{n}")
        self.grid_combo.setCurrentIndex(0)

    def _on_palette_changed(self, name: str):
        """Handle text-based palette changes (for backward compatibility)."""
        if name.startswith("───"):
            return
        self.paletteChanged.emit(name)

    def _on_palette_index_changed(self, index: int):
        """Handle index-based palette changes (for arrow key navigation)."""
        if index < 0:
            return

        name = self.palette_combo.itemText(index)
        if name.startswith("───"):
            return

        self.paletteChanged.emit(name)

    def _emit_offset_changed(self, _v: float):
        self.offsetChanged.emit(self.offset_x_spin.value(), self.offset_y_spin.value())

    def _apply_swatch(
        self,
        button: QPushButton,
        hex_color: str,
    ):
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {hex_color};
                border: 1px solid #656565;
                padding: 4px 10px;
                min-width: 48px;
                color: black;
            }}
            QPushButton:hover {{ border: 1px solid #9a9a9a; }}
            """
        )

    # Setter methods
    def setPlots(
        self,
        labels,
        current_index=0,
    ):
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        for label in labels:
            self.plot_combo.addItem(label)
        if labels:
            idx = min(max(current_index, 0), len(labels) - 1)
        else:
            idx = 0
        self.plot_combo.setCurrentIndex(idx)
        self.plot_combo.blockSignals(False)

    def setAccel(self, value: float):
        self.accel_spin.blockSignals(True)
        self.accel_spin.setValue(value)
        self.accel_spin.blockSignals(False)

    def setPointSize(self, value: float):
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(value)
        self.size_spin.blockSignals(False)

    def setLinesChecked(self, checked: bool):
        self.lines_chk.blockSignals(True)
        self.lines_chk.setChecked(checked)
        self.lines_chk.blockSignals(False)

    # NEW: Setter for visibility checkbox
    def setVisibilityChecked(self, checked: bool):
        self.visible_chk.blockSignals(True)
        self.visible_chk.setChecked(checked)
        self.visible_chk.blockSignals(False)

    def setPaletteEnabled(self, enabled: bool):
        self.palette_combo.setEnabled(enabled)

    def setSelectedPalette(self, name: str):
        idx = self.palette_combo.findText(name)
        if idx >= 0:
            self.palette_combo.blockSignals(True)
            self.palette_combo.setCurrentIndex(idx)
            self.palette_combo.blockSignals(False)

    def setAxesGridColorSwatch(self, hex_color: str):
        self._apply_swatch(self.axes_grid_btn, hex_color)

    def setADCGridColorSwatch(self, hex_color: str):
        self._apply_swatch(self.adc_grid_btn, hex_color)

    def setInfoText(self, text: str):
        self.info_label.setText(text)

    def setOffset(
        self,
        x: float,
        y: float,
    ):
        self.offset_x_spin.blockSignals(True)
        self.offset_y_spin.blockSignals(True)
        self.offset_x_spin.setValue(x)
        self.offset_y_spin.setValue(y)
        self.offset_x_spin.blockSignals(False)
        self.offset_y_spin.blockSignals(False)
