from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import PIL.Image
from PIL.ImageQt import ImageQt


class ImageWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setMinimumSize(200, 200)
        #self.setFixedSize(200, 200)

        self.image = None

        l = QVBoxLayout(self)
        self.setLayout(l)

        self.scroll_area = QScrollArea(self)
        l.addWidget(self.scroll_area)

        self.canvas = ImageDisplayCanvas(self)
        self.scroll_area.setWidget(self.canvas)

        lh = QHBoxLayout()
        l.addLayout(lh)

        self.repeat_input = QSpinBox(self)
        lh.addWidget(self.repeat_input)
        self.repeat_input.setRange(1, 10)
        self.repeat_input.setValue(self.repeat)

        self.repeat_input.setStatusTip(self.tr("repeat"))
        self.repeat_input.valueChanged.connect(self.set_repeat)

        self.zoom_bar = QScrollBar(Qt.Horizontal, self)
        lh.addWidget(self.zoom_bar)
        self.zoom_bar.setStatusTip(self.tr("zoom"))
        self.zoom_bar.setRange(1, 500)
        self.zoom_bar.setValue(self.zoom)
        self.zoom_bar.valueChanged.connect(self.set_zoom)

        b = QPushButton(self.tr("100%"))
        lh.addWidget(b)
        b.clicked.connect(lambda: self.set_zoom(100))

    @property
    def zoom(self):
        return self.canvas.zoom

    @property
    def repeat(self):
        return self.canvas.num_repeat

    def set_zoom(self, z: int):
        self.canvas.set_zoom(z)

    def set_repeat(self, r: int):
        self.canvas.set_repeat(r)

    def set_image(self, img):
        if isinstance(img, PIL.Image.Image):
            img = ImageQt(img)
        self.image = img
        self.canvas.set_image(self.image)


class ImageDisplayCanvas(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.image = None
        self._zoom = 100
        self.num_repeat = 1
        self._size = (10, 10)

    @property
    def zoom(self):
        return self._zoom

    def set_zoom(self, z):
        self._zoom = z
        self.setFixedSize(
            self._size[0] * self.zoom * self.num_repeat,
            self._size[1] * self.zoom * self.num_repeat)
        self.update()

    def set_repeat(self, r : int):
        self.num_repeat = max(1, r)
        self.setFixedSize(
            self._size[0] * self.zoom * self.num_repeat,
            self._size[1] * self.zoom * self.num_repeat)
        self.update()

    def set_image(self, img):
        self.image = img
        self._size = (self.image.width(), self.image.height())
        self.set_zoom(self.zoom)

    def paintEvent(self, e):
        if self.image is None:
            return
        p = QPainter(self)

        p.fillRect(
            0, 0,
            self.image.width() * self.num_repeat * self.zoom,
            self.image.height() * self.num_repeat * self.zoom,
            Qt.black
        )

        t = QTransform()
        t.scale(self.zoom / 100., self.zoom / 100.)

        p.setTransform(t)

        for y in range(self.num_repeat):
            for x in range(self.num_repeat):
                p.drawImage(x*self.image.width(), y*self.image.height(), self.image)
