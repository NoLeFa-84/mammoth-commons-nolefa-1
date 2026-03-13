from PySide6.QtWidgets import QScrollArea
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)
from .step_utils.card_button import CardButton


class ScrollSelector(QWidget):
    def __init__(self, items, specs, on_change, parent=None):
        super().__init__(parent)

        self.on_change = on_change
        self.items = list(items)
        self.specs = dict(specs)
        self.cards = []
        self.selected = self.items[0] if self.items else None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border: none; color:black")

        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        # Create cards
        for name in self.items:
            html = self.specs.get(name, dict()).get("description", "")
            if not html:
                continue
            card = CardButton(name, html)
            card.clicked.connect(self._select)
            self.cards.append(card)
            self.layout.addWidget(card)

        self.layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        if self.items:
            self.set_selected(self.items[0])

    # ----------------------------
    # Selection logic
    # ----------------------------
    def _select(self, name):
        self.selected = name
        for card in self.cards:
            card.setChecked(card.name == name)

        self.on_change(name)

    def set_selected(self, name):
        if self.selected == name:
            return
        self.selected = name
        for card in self.cards:
            card.setChecked(card.name == name)
        self.on_change(name)

    # ----------------------------
    # ComboBox compatibility
    # ----------------------------
    def currentText(self):
        return self.selected

    def itemText(self, index):
        return self.items[index] if 0 <= index < len(self.items) else ""

    def findText(self, text):
        for i, name in enumerate(self.items):
            if name == text:
                return i
        return -1

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.items):
            self.set_selected(self.items[index])

    def clear(self):
        for card in self.cards:
            card.setParent(None)
            card.deleteLater()
        self.cards.clear()
        self.items.clear()
        self.selected = None

    def removeItem(self, index):
        if 0 <= index < len(self.items):
            self.items.pop(index)
            card = self.cards.pop(index)
            card.setParent(None)
            card.deleteLater()
            if self.items:
                self.set_selected(self.items[0])

    def addItem(self, name):
        html = self.specs.get(name, dict()).get("description", "")
        if not html:
            return
        self.items.append(name)
        card = CardButton(name, html)
        card.clicked.connect(self._select)
        self.cards.append(card)
        self.layout.addWidget(card)

        if self.selected is None:
            self.set_selected(name)

    def addItems(self, names):
        for name in names:
            self.addItem(name)
