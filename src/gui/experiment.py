from functools import partial
from queue import PriorityQueue, Empty
from typing import Any

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PIL import Image
from PIL.ImageQt import ImageQt
from torchvision.transforms.functional import to_pil_image

from .image_display import ImageWidget

from ..parameters import parameters_to_yaml, yaml_to_parameters, convert_params, set_parameter_defaults
from .thread_wrapper import ImageTrainingThread


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
        l.addWidget(self.editor)

        bar = QToolBar()
        l.addWidget(bar)
        for id, name in (
                ("train", self.tr("&Train")),
                ("stop", self.tr("&Stop")),
        ):
            b = QToolButton()
            b.setText(name)
            bar.addWidget(b)
            b.clicked.connect(partial(self.slot_tool_button, id))

        l = QVBoxLayout()
        grid.addLayout(l, 0, 1)
        self.image_display = ImageWidget()
        l.addWidget(self.image_display)

        self.log_display = QPlainTextEdit()
        l.addWidget(self.log_display)

        #img = Image.open("/home/bergi/Pictures/bob/Bobdobbs.png")
        #self.image_display.set_image(img)

    def slot_tool_button(self, id: str):
        if id == "train":
            self.trainer.create()
            self.trainer.start_training()
            # self._pass_parameters()
        elif id == "stop":
            self.trainer.stop_training()
            self.trainer.destroy()

    def set_parameters(self, parameters: dict):
        if parameters:
            yaml_text = parameters_to_yaml(parameters)
        else:
            yaml_text = ""
        self.editor.setPlainText(yaml_text)
        self._pass_parameters()

    def halt(self):
        self.trainer.destroy()

    def _pass_parameters(self):
        try:
            parameters = yaml_to_parameters(self.editor.toPlainText())
            params = convert_params(parameters)
            set_parameter_defaults(params)
        except Exception as e:
            print(e)
            return

        params["verbose"] = 1
        params["snapshot_interval"] = 1.
        self.trainer.set_parameters(params)

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

