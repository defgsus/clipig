import sys
from PyQt5 import QtWidgets

from .main import MainWindow
from ..parameters import parse_arguments

def run_app():
    app = QtWidgets.QApplication(sys.argv)

    parameters = parse_arguments(gui_mode=True)

    window = MainWindow()
    window.show()
    window.create_new_experiment(parameters=parameters)

    sys.exit(app.exec())

