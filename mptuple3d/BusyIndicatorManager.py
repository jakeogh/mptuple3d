#!/usr/bin/env python3
# tab-width:4

# pylint: disable=no-name-in-module
from __future__ import annotations

import time
from contextlib import contextmanager

from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QLabel


def timestamp():
    """High resolution timestamp for logging."""
    return f"{time.time():.6f}"


class BusyIndicatorManager:
    """
    Manages busy state visual indicators for the viewer.

    STRICT VERSION: Fails loudly when status label is not properly connected.
    No more silent failures that hide real problems!
    """

    def __init__(self, status_label: None | QLabel = None):
        """
        Initialize busy indicator manager.

        Args:
            status_label: Optional QLabel widget to use as status indicator
        """
        self.status_label = status_label
        self.is_busy = False
        self.busy_count = 0  # Track nested busy operations

        # Timer to delay busy indicator to avoid flicker on very fast operations
        self.busy_timer = QTimer()
        self.busy_timer.setSingleShot(True)
        self.busy_timer.timeout.connect(self._show_busy_immediate)
        self.busy_delay_ms = 0  # FIXED: Show immediately, no delay

        # Timer to ensure minimum busy display time for visibility
        self.min_busy_timer = QTimer()
        self.min_busy_timer.setSingleShot(True)
        self.min_busy_timer.timeout.connect(self._hide_busy_immediate)
        self.min_busy_time_ms = (
            1000  # INCREASED: Keep busy visible for 1 second minimum
        )

        # State tracking
        self._pending_hide = False

        # NUCLEAR OPTION: Use QPalette instead of stylesheets to bypass CSS
        self.original_palette = None

    def set_status_label(self, label: QLabel) -> None:
        """Set the status label widget - REQUIRED for operations."""
        if label is None:
            raise ValueError("status_label cannot be None!")
        if not isinstance(label, QLabel):
            raise TypeError(f"status_label must be QLabel, got {type(label).__name__}")

        self.status_label = label
        self.original_palette = label.palette()  # Store original palette
        self._apply_idle_style()

    def _require_status_label(self) -> None:
        """Internal method to ensure status label is connected before operations."""
        if self.status_label is None:
            raise RuntimeError(
                "BusyIndicatorManager: No status label connected! "
                "Call set_status_label() before using busy operations. "
                "This is a programming error that must be fixed."
            )

    @contextmanager
    def busy_operation(self, operation_name: str = "Processing"):
        """
        Context manager for operations that should show busy state.

        Args:
            operation_name: Name of the operation for debugging

        Usage:
            with busy_manager.busy_operation("Updating plot"):
                # Do expensive operation
                update_plot()
        """
        self.start_busy(operation_name)
        try:
            yield
        finally:
            self.end_busy(operation_name)

    def start_busy(self, operation_name: str = "Processing") -> None:
        """Start a busy operation (non-context manager API)."""
        self._require_status_label()  # FAIL LOUDLY if not connected
        self._start_busy(operation_name)

    def end_busy(self, operation_name: str = "Processing") -> None:
        """End a busy operation (non-context manager API)."""
        self._require_status_label()  # FAIL LOUDLY if not connected
        self._end_busy(operation_name)

    def _start_busy(self, operation_name: str) -> None:
        """Start a busy operation."""
        self.busy_count += 1

        if self.busy_count == 1:  # First busy operation
            self._pending_hide = False

            if not self.is_busy:
                if self.busy_delay_ms == 0:
                    self._show_busy_immediate()
                else:
                    self.busy_timer.start(self.busy_delay_ms)

    def _end_busy(self, operation_name: str) -> None:
        """End a busy operation."""
        self.busy_count = max(0, self.busy_count - 1)

        if self.busy_count == 0:  # Last busy operation finished
            # Cancel the delay timer if operation finished before delay
            self.busy_timer.stop()

            if self.is_busy:
                # If we're currently showing busy, ensure minimum display time
                self._pending_hide = True
                if not self.min_busy_timer.isActive():
                    # Start minimum display timer
                    self.min_busy_timer.start(self.min_busy_time_ms)
            else:
                #print(f"{timestamp()} [BUSY] Operation completed before busy was shown")
                pass

    def _show_busy_immediate(self) -> None:
        """Show busy indicator immediately."""
        if self.busy_count > 0 and not self.is_busy:
            self.is_busy = True
            self._apply_busy_style()
            #print(
            #    f"{timestamp()} [BUSY] *** INDICATOR SHOWN *** (should be BLACK with BUSY text)"
            #)

    def _hide_busy_immediate(self) -> None:
        """Hide busy indicator immediately."""
        if self._pending_hide:
            self.is_busy = False
            self._pending_hide = False
            self._apply_idle_style()
            #print(
            #    f"{timestamp()} [BUSY] *** INDICATOR HIDDEN *** (should be normal now)"
            #)

    def _apply_busy_style(self) -> None:
        """Apply busy visual style using QPalette (bypasses CSS)."""
        self._require_status_label()  # FAIL LOUDLY

        #print(f"{timestamp()} [BUSY] Applying BLACK busy style...")

        # Method 1: Use QPalette to bypass CSS completely
        busy_palette = QPalette()
        busy_palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        busy_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        busy_palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)

        self.status_label.setText("BUSY")
        self.status_label.setPalette(busy_palette)
        self.status_label.setAutoFillBackground(True)

        # Method 2: Also try stylesheet with maximum specificity
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #000000 !important;
                color: #ffffff !important;
                border: 2px solid #333333 !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
                font-weight: bold !important;
                font-size: 10px !important;
            }
        """
        )

        self.status_label.update()
        self.status_label.repaint()
        #print(f"{timestamp()} [BUSY] Style applied to label: '{self.status_label.text()}'")

    def _apply_idle_style(self) -> None:
        """Apply idle visual style."""
        self._require_status_label()  # FAIL LOUDLY

        #print(f"{timestamp()} [BUSY] Applying normal idle style...")

        # Restore original palette
        if self.original_palette:
            self.status_label.setPalette(self.original_palette)

        self.status_label.setAutoFillBackground(False)
        self.status_label.setText("")  # Clear text when idle

        # Clear any custom stylesheet
        self.status_label.setStyleSheet("")

        self.status_label.update()
        self.status_label.repaint()
        #print(f"{timestamp()} [BUSY] Idle style applied")

    def force_busy(self, show: bool = True) -> None:
        """
        Force busy state on/off (for testing or special cases).

        Args:
            show: True to show busy, False to hide
        """
        self._require_status_label()  # FAIL LOUDLY
        #print(f"{timestamp()} [BUSY] FORCE_BUSY: {show}")

        if show:
            self.is_busy = True
            self._apply_busy_style()
        else:
            self.is_busy = False
            self._pending_hide = False
            self.busy_count = 0
            self.busy_timer.stop()
            self.min_busy_timer.stop()
            self._apply_idle_style()

    def set_busy_delay(self, delay_ms: int) -> None:
        """Set the delay before showing busy indicator."""
        self.busy_delay_ms = max(0, delay_ms)

    def set_min_busy_time(self, time_ms: int) -> None:
        """Set the minimum time to show busy indicator."""
        self.min_busy_time_ms = max(0, time_ms)

    def get_status(self) -> dict:
        """Get current busy status for debugging."""
        return {
            "is_busy": self.is_busy,
            "busy_count": self.busy_count,
            "pending_hide": self._pending_hide,
            "delay_timer_active": self.busy_timer.isActive(),
            "min_timer_active": self.min_busy_timer.isActive(),
            "has_status_label": self.status_label is not None,
            "status_label_id": id(self.status_label) if self.status_label else None,
            "label_text": self.status_label.text() if self.status_label else None,
        }
