#!/usr/bin/env python3
# tab-width:4

STYLE = """
    QWidget {
        background-color: #353535;
        color: black;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 11px;
    }

    /* Labels - CHANGE COLOR HERE IN ONE SPOT */
    QLabel {
        color: black;  /* <-- Change this to any color you want for ALL labels */
        background-color: transparent;
        border: none;
        padding: 2px 4px;
        font-weight: normal;
    }

    /* Buttons - raised appearance with gradient */
    QPushButton {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #5a5a5a, stop: 1 #454545);
        border: 1px solid #707070;
        border-radius: 3px;
        padding: 4px 8px;
        color: black;
        font-weight: bold;
        min-width: 50px;
    }

    QPushButton:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #6a6a6a, stop: 1 #555555);
        border: 1px solid #808080;
    }

    QPushButton:pressed {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #404040, stop: 1 #505050);
        border: 1px solid #606060;
    }

    /* Spin boxes - inset appearance */
    QDoubleSpinBox {
        background-color: #3a3a3a;
        border: 2px inset #656565;
        border-radius: 2px;
        padding: 2px 4px;
        color: black;
        selection-background-color: #4a90e2;
    }

    QDoubleSpinBox:focus {
        border: 2px inset #4a90e2;
    }

    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        background-color: #505050;
        border: 1px solid #707070;
        width: 16px;
    }

    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #606060;
    }

    /* Combo boxes - dropdown appearance */
    QComboBox {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #505050, stop: 1 #404040);
        border: 1px solid #707070;
        border-radius: 3px;
        padding: 3px 20px 3px 6px;
        color: black;
        min-width: 80px;
    }

    QComboBox:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #606060, stop: 1 #505050);
        border: 1px solid #808080;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid #707070;
        background-color: #505050;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }

    QComboBox::down-arrow {
        image: none;
        border: 2px solid #cccccc;
        border-top: none;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        width: 0px;
        height: 0px;
    }

    QComboBox QAbstractItemView {
        background-color: #404040;
        border: 1px solid #707070;
        selection-background-color: #4a90e2;
        selection-color: white;
        outline: none;
    }

    /* Checkboxes - distinct appearance */
    QCheckBox {
        spacing: 8px;
        color: black;
        background-color: transparent;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #707070;
        border-radius: 2px;
        background-color: #3a3a3a;
    }

    QCheckBox::indicator:hover {
        border: 2px solid #808080;
        background-color: #4a4a4a;
    }

    QCheckBox::indicator:checked {
        background-color: #4a90e2;
        border: 2px solid #5aa0f2;
    }

    QCheckBox::indicator:checked:hover {
        background-color: #5aa0f2;
    }
    """
