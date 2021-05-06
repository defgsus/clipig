from functools import partial
from queue import PriorityQueue, Empty
from typing import Any

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Statistics(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = []
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)

        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)

    def add_stats(self, stats: dict):
        self._stats.append(stats)

        self.pbar.setMaximum(stats["epochs"])
        self.pbar.setValue(stats["epoch"])
