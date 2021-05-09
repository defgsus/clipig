from typing import Any, Optional

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

        l = QHBoxLayout()
        layout.addLayout(l)
        self.label_framerate = QLabel()
        l.addWidget(self.label_framerate)

        self.label_time = QLabel()
        l.addWidget(self.label_time)

    def stats(self) -> Optional[dict]:
        return self._stats[-1] if self._stats else None

    def add_stats(self, stats: dict):
        self._stats.append(stats)

        self.pbar.setMaximum(stats["epochs"])
        self.pbar.setValue(stats["epoch"])
        
        e_per_s = 1. / stats['average_frame_rate'] if stats['average_frame_rate'] else 0.
        self.label_framerate.setText(f"{e_per_s:.2f} epochs/s")
        self.label_time.setText(
            f"train time: {stats['training_seconds']:.2f}s"
            f" (fwd {stats['forward_seconds']:.2f}s"
            f", backwd {stats['backward_seconds']:.2f}s)"
        )
