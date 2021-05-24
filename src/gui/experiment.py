from copy import deepcopy
from functools import partial
from queue import Queue, Empty
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
        self._queue = Queue()
        self.trainer = ImageTrainingThread(self._queue)
        self._training_parameters = None
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
        self.log_display.setMaximumHeight(200)
        font = QFont("Mono")
        font.setPointSize(7)
        self.log_display.setFont(font)
        l.addWidget(self.log_display)
        #grid.addWidget(self.log_display, 2, 0, 1, 2)

        bar = QToolBar()
        l.addWidget(bar)
        self.tool_buttons = dict()
        for id, name in (
                ("start", self.tr("&Start")),
                ("update", self.tr("&Update")),
                ("stop", self.tr("Sto&p")),
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
        """
        Return the currently trained parameters
        """
        if self._training_parameters:
            params = deepcopy(self._training_parameters)
        else:
            params = self._get_parameters(for_training=False)
        if params:
            for key in (
                    "verbose", "output", "snapshot_interval",
                    "start_epoch", "device"
            ):
                params.pop(key, None)
        return params

    def get_image(self) -> Optional[QImage]:
        return self.image_display.image

    def get_config_header(self) -> str:
        stats = self.trainer.running_counts()
        if stats:
            run_time = stats['training_seconds']
            epoch = stats['training_epochs']
        else:
            run_time = self.trainer._trainer.training_seconds
            epoch = self.trainer._trainer.epoch
        return self.trainer._trainer.get_config_header(run_time=run_time, epoch=epoch)

    def slot_tool_button(self, id: str):
        # print("SLOT", id)
        parameters = self._get_parameters()

        if id == "start" and parameters:
            self.trainer.create()
            self.trainer.start_training(parameters)
            self._training_parameters = deepcopy(parameters)

        elif id == "update" and parameters:
            self.trainer.create()
            self.trainer.update_parameters(parameters)
            self._training_parameters = deepcopy(parameters)

        elif id == "stop":
            self.trainer.pause_training()

    def _process_queue_message(self, name: str, data: Any):
        # print("FROM TRAINER:", name, data)
        if name == "snapshot":
            image = to_pil_image(data)
            self.image_display.set_image(image)

        elif name == "log":
            self.log_display.appendPlainText(data)

        elif name == "progress":
            self._add_stats(data)

        elif name == "started":
            pass

        elif name == "stopped":
            pass

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
        for i in range(20):
            try:
                message = self._queue.get_nowait()
                self._process_queue_message(message.name, message.data)
            except Empty:
                break

        self._call_queue_processing()

    def _add_stats(self, stats: dict):
        self.statistics.add_stats(stats)
