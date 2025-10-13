import os

os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

from PySide6.QtWidgets import QApplication, QMainWindow
import sys
from mai_bias.states.dashboard import Dashboard
from mai_bias.states.step import load_all_runs
from mai_bias.states.steps.dataset import SelectDataset
from mai_bias.states.steps.model import SelectModel
from mai_bias.states.steps.analysis import SelectAnalysis
from mai_bias.states.results import Results
from mai_bias.backend.loaders import registry
from PySide6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect
from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QPoint,
    Signal,
)
from PySide6.QtGui import QColor

"""from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWebEngineCore import QWebEngineSettings
fmt = QSurfaceFormat()
fmt.setVersion(3, 3)  # OpenGL 3.3 or higher
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)
"""

items = load_all_runs("history.json")


class SlidingStackedWidget(QStackedWidget):
    """
    This class (and only this class) is a modified version of third-party cpp code.
    The original code was governed by the following license.

    MIT License
    Copyright (c) 2020 Tim Schneeberger (ThePBone) <tim.schneeberger(at)outlook.de>
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    animationFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_mainwindow = parent if parent else self
        self.m_vertical = False
        self.m_speed = 500
        self.m_animationtype = QEasingCurve.Type.InOutQuart
        self.m_now = 0
        self.m_next = 0
        self.m_wrap = False
        self.m_pnow = QPoint(0, 0)
        self.m_active = False
        self.invert = False

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#ffffff"))
        self.setPalette(palette)

    def slideInNext(self) -> bool:
        now = self.currentIndex()
        if self.m_wrap or now < self.count() - 1:
            self.slideInIdx(now + 1)
            return True
        return False

    def slideInPrev(self) -> bool:
        now = self.currentIndex()
        if self.m_wrap or now > 0:
            self.slideInIdx(now - 1)
            return True
        return False

    def slideInIdx(self, idx: int):
        if idx >= self.count():
            idx %= self.count()
        elif idx < 0:
            idx = (idx + self.count()) % self.count()
        self.slideInWgt(self.widget(idx))

    def slideInWgt(self, new_widget: QWidget):
        if self.m_active:
            return
        self.m_active = True
        now = self.currentIndex()
        next_idx = self.indexOf(new_widget)
        if now == next_idx:
            self.m_active = False
            return

        offset_x = self.frameRect().width()
        offset_y = self.frameRect().height()
        new_widget.setGeometry(0, 0, offset_x, offset_y)

        direction = (
            -offset_x if not self.m_vertical else 0,
            -offset_y if self.m_vertical else 0,
        )
        if self.invert:
            direction = (-direction[0], -direction[1])
        pnext = new_widget.pos()
        pnow = self.widget(now).pos()
        self.m_pnow = pnow
        new_widget.move(pnext.x() - direction[0], pnext.y() - direction[1])

        new_widget.show()
        QApplication.processEvents()
        new_widget.raise_()

        anim_now = QPropertyAnimation(self.widget(now), b"pos")
        anim_now.setDuration(self.m_speed)
        anim_now.setEasingCurve(self.m_animationtype)
        anim_now.setStartValue(QPoint(pnow.x(), pnow.y()))
        anim_now.setEndValue(QPoint(direction[0] + pnow.x(), direction[1] + pnow.y()))

        anim_now_op_eff = QGraphicsOpacityEffect()
        self.widget(now).setGraphicsEffect(anim_now_op_eff)
        anim_now_op = QPropertyAnimation(anim_now_op_eff, b"opacity")
        anim_now_op.setDuration(self.m_speed)
        anim_now_op.setStartValue(1)
        anim_now_op.setEndValue(0)
        anim_now_op.finished.connect(lambda: anim_now_op_eff.deleteLater())

        anim_next_op_eff = QGraphicsOpacityEffect()
        anim_next_op_eff.setOpacity(0)
        new_widget.setGraphicsEffect(anim_next_op_eff)
        anim_next_op = QPropertyAnimation(anim_next_op_eff, b"opacity")
        anim_next_op.setDuration(self.m_speed)
        anim_next_op.setStartValue(0)
        anim_next_op.setEndValue(1)
        anim_next_op.finished.connect(lambda: anim_next_op_eff.deleteLater())

        anim_next = QPropertyAnimation(new_widget, b"pos")
        anim_next.setDuration(self.m_speed)
        anim_next.setEasingCurve(self.m_animationtype)
        anim_next.setStartValue(
            QPoint(-direction[0] + pnext.x(), direction[1] + pnext.y())
        )
        anim_next.setEndValue(QPoint(pnext.x(), pnext.y()))

        self.animgroup = QParallelAnimationGroup()
        self.animgroup.addAnimation(anim_now)
        self.animgroup.addAnimation(anim_next)
        # self.animgroup.addAnimation(anim_now_op)
        # self.animgroup.addAnimation(anim_next_op)
        self.animgroup.finished.connect(self.animationDoneSlot)

        self.m_next = next_idx
        self.m_now = now
        self.m_active = True
        self.animgroup.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def animationDoneSlot(self):
        self.setCurrentIndex(self.m_next)
        self.widget(self.m_now).hide()
        self.widget(self.m_now).move(self.m_pnow)
        self.m_active = False
        self.animationFinished.emit()

    def slideToWidget(self, index):
        if index == self.currentIndex():
            return
        self.invert = self.currentIndex() > index
        self.slideInIdx(index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        tags = {
            key: "<h1>" + key + "</h1>" + module["description"]
            for key, module in (
                registry.dataset_loaders
                | registry.model_loaders
                | registry.analysis_methods
            ).items()
        }
        self.setWindowTitle("MAI-BIAS local runner")
        self.setGeometry(100, 100, 1200, 900)
        self.stacked_widget = SlidingStackedWidget()
        active_run = [None]
        self.stacked_widget.addWidget(
            Dashboard(self.stacked_widget, items, tags, active_run)
        )
        self.stacked_widget.addWidget(
            SelectDataset(
                "Data", self.stacked_widget, registry.dataset_loaders, active_run, items
            )
        )
        self.stacked_widget.addWidget(
            SelectModel(
                "Model", self.stacked_widget, registry.model_loaders, active_run, items
            )
        )
        self.stacked_widget.addWidget(
            SelectAnalysis(
                "Analysis method",
                self.stacked_widget,
                registry.analysis_methods,
                active_run,
                items,
            )
        )
        self.stacked_widget.addWidget(
            Results(self.stacked_widget, active_run, tags, items)
        )
        self.setCentralWidget(self.stacked_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
