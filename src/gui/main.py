from typing import Optional, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .experiment import Experiment


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(self.tr("CLIPig"))

        self._create_main_menu()
        self._create_widgets()

        self.setGeometry(0, 0, 800, 600)

    def _create_main_menu(self):
        menu = self.menuBar().addMenu(self.tr("&File"))

        new_menu = menu.addMenu(self.tr("New"))
        new_menu.addAction(self.tr("&experiment"), self.slot_new_experiment, QKeySequence("Ctrl+N"))

        menu.addAction(self.tr("E&xit"), self.close)

    def _create_widgets(self):
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.slot_tab_changed)

    def experiments(self) -> List[Experiment]:
        return [
            self.tab_widget.widget(i)
            for i in range(self.tab_widget.count())
            if isinstance(self.tab_widget.widget(i), Experiment)
        ]

    def closeEvent(self, event: QCloseEvent) -> None:
        for e in self.experiments():
            e.halt()

    def slot_new_experiment(self):
        widget = Experiment()
        widget.tab = self.tab_widget.addTab(widget, self.tr("experiment"))

    def slot_tab_changed(self):
        widget = self.tab_widget.currentWidget()

    def create_new_experiment(
            self,
            parameters: Optional[dict] = None
    ):
        widget = Experiment()
        widget.tab = self.tab_widget.addTab(widget, self.tr("experiment"))
        widget.set_parameters(parameters)
