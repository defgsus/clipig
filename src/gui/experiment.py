from functools import partial
from queue import PriorityQueue, Empty
from typing import Any, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PIL import Image
from PIL.ImageQt import ImageQt
from torchvision.transforms.functional import to_pil_image

from .image_display import ImageWidget

from ..parameters import parameters_to_yaml, yaml_to_parameters, convert_params, set_parameter_defaults
from .thread_wrapper import ImageTrainingThread
from .statistics import Statistics


class Experiment(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = PriorityQueue()
        self.trainer = ImageTrainingThread(self._queue)
        self._create_widgets()
        self._call_queue_processing()

    def _create_widgets(self):
        grid = QGridLayout(self)

        l = QVBoxLayout()
        grid.addLayout(l, 0, 0)
        self.editor = QPlainTextEdit()
        font = QFont("Mono")
        font.setPointSize(17)
        self.editor.setFont(font)
        l.addWidget(self.editor)

        self.log_display = QPlainTextEdit()
        self.log_display.setMaximumHeight(100)
        l.addWidget(self.log_display)
        #grid.addWidget(self.log_display, 2, 0, 1, 2)

        bar = QToolBar()
        l.addWidget(bar)
        self.tool_buttons = dict()
        for id, name in (
                ("start", self.tr("&Start")),
                ("update", self.tr("&Update")),
                ("pause", self.tr("&Pause")),
        ):
            b = QToolButton()
            b.setText(name)
            bar.addWidget(b)
            b.clicked.connect(partial(self.slot_tool_button, id))
            self.tool_buttons[id] = b

        l = QVBoxLayout()
        grid.addLayout(l, 0, 1)
        self.image_display = ImageWidget()
        l.addWidget(self.image_display)

        self.statistics = Statistics()
        grid.addWidget(self.statistics, 1, 0, 1, 2)

    def get_parameters(self) -> Optional[dict]:
        return self._get_parameters(for_training=False)

    def get_image(self) -> Optional[QImage]:
        return self.image_display.image

    def slot_tool_button(self, id: str):
        parameters = self._get_parameters()

        if id == "start" and parameters:
            self.trainer.create()
            self.trainer.start_training(parameters)

        elif id == "update" and parameters:
            self.trainer.update_parameters(parameters)

        elif id == "pause":
            b: QToolButton = self.tool_buttons["pause"]
            if b.isDown():
                b.setDown(False)
                self.trainer.continue_training()
            else:
                b.setDown(True)
                self.trainer.pause_training()

    def set_parameters(self, parameters: dict):
        if parameters:
            yaml_text = parameters_to_yaml(parameters)
        else:
            yaml_text = ""
        self.editor.setPlainText(yaml_text)

    def halt(self):
        self.trainer.destroy()

    def _get_parameters(self, for_training: bool = True) -> Optional[dict]:
        try:
            parameters = yaml_to_parameters(self.editor.toPlainText())
            params = convert_params(parameters)
            set_parameter_defaults(params)
        except Exception as e:
            print(e)
            return

        if for_training:
            params["verbose"] = 2
            params["snapshot_interval"] = 1.
        return params

    def _call_queue_processing(self):
        QTimer.singleShot(100, self._process_queue)

    def _process_queue(self):
        try:
            message = self._queue.get_nowait()
            self._process_queue_message(message.name, message.data)
        except Empty:
            pass

        self._call_queue_processing()

    def _process_queue_message(self, name: str, data: Any):
        # print("queue:", name, data)
        if name == "snapshot":
            image = to_pil_image(data)
            self.image_display.set_image(image)

        elif name == "log":
            self.log_display.appendPlainText(data)

        elif name == "progress":
            self._add_stats(data)

    def _add_stats(self, stats: dict):
        self.statistics.add_stats(stats)
