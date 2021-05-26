from copy import deepcopy
from typing import Any, Optional, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..parameters import parameters_to_yaml, yaml_to_parameters, convert_params, set_parameter_defaults


class YamlEditor(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._font_size = 17
        self._create_widgets()
        self.set_font_size(self._font_size)

    def _create_widgets(self):
        l = QVBoxLayout(self)

        self.editor = QPlainTextEdit()
        l.addWidget(self.editor)

        self.error_label = QLabel()
        l.addWidget(self.error_label)

    def create_actions(self, menu: QMenu):
        menu.addAction(
            self.tr("zoom in"),
            lambda: self.set_font_size(self._font_size + 1),
            QKeySequence("Ctrl++"),
        )
        menu.addAction(
            self.tr("zoom in"),
            lambda: self.set_font_size(self._font_size - 1) if self._font_size > 1 else None,
            QKeySequence("Ctrl+-"),
        )

    def set_font_size(self, size: int):
        self._font_size = size
        font = QFont("Mono")
        font.setPointSize(self._font_size)
        self.editor.setFont(font)

    def set_parameters(self, parameters: Union[str, dict]):
        self.clear_error()

        if isinstance(parameters, dict):
            yaml_text = parameters_to_yaml(parameters)
        else:
            yaml_text = parameters
        self.editor.setPlainText(yaml_text)

    def get_parameters(self) -> Optional[dict]:
        self.clear_error()

        try:
            parameters = yaml_to_parameters(self.editor.toPlainText())
            params = convert_params(parameters)
            set_parameter_defaults(params)
            return params

        except Exception as e:
            self.set_error(f"{type(e).__name__}: {e}")
            return

    def clear_error(self):
        self.error_label.setText("")

    def set_error(self, text: str):
        self.error_label.setText(text)
