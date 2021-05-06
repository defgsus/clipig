from typing import Optional, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .experiment import Experiment
from ..parameters import save_yaml_config


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_filename = None

        self.setWindowTitle(self.tr("CLIPig"))

        self._create_main_menu()
        self._create_widgets()

        self.setGeometry(0, 0, 800, 600)

    def _create_main_menu(self):
        menu = self.menuBar().addMenu(self.tr("&File"))

        menu.addAction(self.tr("new &experiment"), self.slot_new_experiment, QKeySequence("Ctrl+N"))
        menu.addAction(self.tr("save image"), self.slot_save_image, QKeySequence("Ctrl+S"))

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

    def experiment(self) -> Optional[Experiment]:
        if not self.tab_widget.count():
            return None
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, Experiment):
            return widget

    def closeEvent(self, event: QCloseEvent) -> None:
        for e in self.experiments():
            e.halt()

    def slot_new_experiment(self):
        widget = Experiment()
        widget.tab = self.tab_widget.addTab(widget, self.tr("experiment"))

    def slot_tab_changed(self):
        widget = self.tab_widget.currentWidget()
        # TODO: update menu

    def slot_save_image(self):
        exp = self.experiment()
        if exp:
            self.save_image(exp)

    def create_new_experiment(
            self,
            parameters: Optional[dict] = None
    ):
        widget = Experiment()
        widget.tab = self.tab_widget.addTab(widget, self.tr("experiment"))
        if parameters:
            parameters = parameters.copy()
            parameters.pop("output", None)
            widget.set_parameters(parameters)

    def save_image(self, experiment: Experiment):
        image = experiment.get_image()
        if not image:
            return
        parameters = experiment.get_parameters()

        init_filename = self._last_filename
        if init_filename and init_filename.lower().endswith(".png"):
            init_filename = init_filename[:-4]
        filename, filter = QFileDialog.getSaveFileName(
            self, self.tr("Save image"),
            filter="*.png",
            directory=init_filename,
        )
        if not filename:
            return
        if filename.lower().endswith(".png"):
            filename = filename[:-4]
        self._last_filename = filename

        image.save(filename + ".png")
        if parameters:
            save_yaml_config(
                filename + ".yaml", parameters,
                header="""# saved by CLIPig-gui\n""",
            )
