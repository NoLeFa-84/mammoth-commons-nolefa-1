# -------------------------------------------------------------
#  CardButton – static size, info‑button → white popup, checked colour
# -------------------------------------------------------------
from mammoth_commons.externals import prepare_html
from mammoth_commons.exports import get_description_header

from PySide6.QtCore import Qt, Signal, QUrl, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QLabel,
    QPushButton,
    QDialog,
    QDialogButtonBox,
)


class CardButton(QFrame):
    """
    A compact card that never changes its height.

    • Title is always visible.
    • An ℹ‑button opens a *white* “Module info” dialog with the full HTML.
    • The widget can be marked *checked* – the background becomes #d0d0d0.
    • Emits ``clicked(str)`` when the title area (not the info button) is pressed.
    """

    clicked = Signal(str)

    # -----------------------------------------------------------------
    #  Dark‑background CSS fragment used by the popup (kept from original)
    # -----------------------------------------------------------------
    HTML_BG_DARK = """
    <style>
        html, body {
            background-color: #d0d0d0 !important;
            margin: 0;
            padding: 0;
        }
    </style>
    """

    # -----------------------------------------------------------------
    def __init__(self, name: str, html_description: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.full_html = html_description
        self._checked = False
        self.setObjectName("CardFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame#CardFrame {
                border: 0px solid white;
                border-radius: 8px;
                padding: 5px;
                padding-left: 15px;
                background-color: transparent;
            }
            QFrame#CardFrame:hover {
                background-color: #eeeeee;
            }
            QFrame#CardFrame[checked="true"] {
                background-color: #d0d0d0;   /* colour used for a selected card */
            }
            """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        self.title_label = QLabel(self)
        self.title_label.setTextFormat(Qt.TextFormat.RichText)
        self.title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.title_label.setOpenExternalLinks(False)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size: 22px; border: none;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Build the compact header – same algorithm that originally produced it
        def _fix_img_styles(html: str) -> str:
            import re

            html = re.sub(
                r'<img([^>]+)style="[^"]*height:\s*(\d+)px[^"]*"([^>]*)>',
                r'<img\1height="28"\3>',
                html,
            )
            html = re.sub(
                r'<img([^>]+)style="[^"]*width:\s*(\d+)px[^"]*"([^>]*)>',
                r'<img\1width="\2"\3>',
                html,
            )
            return html

        header_html = prepare_html(get_description_header(self.full_html))
        for tag in ("h1", "h2", "h3"):
            header_html = header_html.replace(f"<{tag}>", " ").replace(f"</{tag}>", " ")
        header_html = _fix_img_styles(header_html)
        header_html = (
            f'<table cellpadding="0" cellspacing="0" style="border:0;"><tr>'
            f'<td style="vertical-align:middle;">{header_html}</td>'
            f"</tr></table>"
        )
        self.title_label.setText(header_html)
        self.info_btn = QPushButton(self)
        self.info_btn.setToolTip("Show module information")
        self.info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.info_btn.setFixedSize(QSize(24, 24))
        self.info_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                font-size: 18px;
            }
            QPushButton:hover {
                color: #5555ff;
            }
            """)
        self.info_btn.setText("\u2139")
        title_row.addWidget(self.title_label, 1)  # stretch → fill available width
        title_row.addWidget(self.info_btn)
        main_layout.addLayout(title_row)
        self.title_label.mousePressEvent = lambda e: self.clicked.emit(self.name)
        self.info_btn.clicked.connect(self._show_popup)

    def _show_popup(self):
        dlg = QDialog(self, Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        dlg.setParent(None)
        dlg.setWindowTitle("Module info")
        dlg.resize(640, 480)
        dlg.setStyleSheet("QDialog { background-color: white; }")
        dlg.setAutoFillBackground(True)
        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(8, 8, 8, 8)
        web = QWebEngineView(dlg)
        web.setZoomFactor(0.9)
        web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        # Build HTML – we want a **white** page, so the body background is overridden.
        html = prepare_html(f"""
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
                  rel="stylesheet">
            <style>
                html, body {{ background-color: white !important; margin:0; padding:0; }}
            </style>
            {self.full_html}
            """)
        web.setHtml(html, QUrl("file:///"))
        vbox.addWidget(web)
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        btn_box.rejected.connect(dlg.reject)
        vbox.addWidget(btn_box)
        dlg.exec()

    def setChecked(self, checked: bool):
        """Mark the card as checked / unchecked."""
        if self._checked == checked:
            return  # no work needed

        self._checked = checked
        self.setProperty("checked", "true" if checked else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def isChecked(self) -> bool:
        return self._checked

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.info_btn.geometry().contains(event.pos()):
            self.clicked.emit(self.name)
