from mammoth_commons.externals import prepare_html
from mammoth_commons.exports import get_description_header
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QSizePolicy,
    QLabel,
)


class CardButton(QFrame):
    clicked = Signal(str)

    HTML_BG_DARK = """
    <style>
        html, body {
            background-color: #d0d0d0 !important;
            margin: 0;
            padding: 0;
        }
    </style>
    """

    def __init__(self, name, html_description, parent=None):
        super().__init__(parent)

        self.name = name
        self.full_html = html_description
        self._checked = False

        self.setObjectName("CardFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame#CardFrame {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 5px;
                padding-left: 15px;
            }
            QFrame#CardFrame[checked="true"] {
                background-color: #d0d0d0;
                border-color: #999999;
            }
            QFrame:hover {
                background-color: #ffffff;
            }
            """)

        # ------------------------------------------------------
        # Main layout
        # ------------------------------------------------------
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # ------------------------------------------------------
        # Title (collapsed view)
        # ------------------------------------------------------
        self.title_label = QLabel(self)
        self.title_label.setTextFormat(Qt.TextFormat.RichText)
        self.title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.title_label.setOpenExternalLinks(False)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size: 22px; border: none;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.setMouseTracking(True)
        self.title_label.setMouseTracking(True)

        def fix_img_styles_for_qlabel(html: str) -> str:
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
        header_html = header_html.replace("<h1>", " ").replace("</h1>", " ")
        header_html = header_html.replace("<h2>", " ").replace("</h2>", " ")
        header_html = header_html.replace("<h3>", " ").replace("</h3>", " ")
        header_html = fix_img_styles_for_qlabel(header_html)
        header_html = (
            f'<table cellpadding="0" cellspacing="0" style="border:0;"><tr>'
            f'<td style="vertical-align:middle;">{header_html}</td>'
            f"</tr></table>"
        )
        self.title_label.setText(header_html)
        self.title_label.mousePressEvent = lambda e: self.clicked.emit(self.name)

        # ------------------------------------------------------
        # WebEngine (expanded view)
        # ------------------------------------------------------
        self.web = QWebEngineView(self)
        self.web.setZoomFactor(0.8)
        self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._load_html(checked=False)

        self.web.hide()
        self.title_label.setFixedHeight(28)

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.web)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _load_html(self, checked: bool):
        bg = self.HTML_BG_DARK
        html = prepare_html("""
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
                  rel="stylesheet">
            """ + bg + self.full_html)
        self.web.setHtml(html, QUrl("file:///"))

    def enterEvent(self, event):
        if not self._checked:
            self.setStyleSheet("""
                QFrame#CardFrame {
                    background-color: #d0d0d0;
                    border: 1px solid #cccccc;
                    border-radius: 8px;
                    padding: 5px;
                    padding-left: 15px;
                }
            """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._checked:
            self.setStyleSheet("""
                QFrame#CardFrame {
                    background-color: #f5f5f5;
                    border: 1px solid #cccccc;
                    border-radius: 8px;
                    padding: 5px;
                    padding-left: 15px;
                }
                QFrame#CardFrame[checked="true"] {
                    background-color: #e0e0e0;
                    border-color: #999999;
                }
            """)
        super().leaveEvent(event)

    def setChecked(self, checked: bool):
        self._checked = checked
        self.setProperty("checked", "true" if checked else "false")

        # force style refresh
        self.style().unpolish(self)
        self.style().polish(self)

        if checked:
            # selected card: show expanded content
            self.title_label.hide()
            self.web.show()
            self.web.setFixedHeight(
                28 + 12 * len(self.full_html.split("<details>")[0].split("\n"))
            )

            # ensure selected card uses selected background, not hover
            self.setStyleSheet("""
                QFrame#CardFrame[checked="true"] {
                    background-color: #d0d0d0;
                    border: 1px solid #999999;
                    border-radius: 8px;
                    padding: 5px;
                    padding-left: 15px;
                }
            """)

        else:
            # unselected card: reset to base (non-hover) style
            self.setStyleSheet("""
                QFrame#CardFrame {
                    background-color: #f5f5f5;
                    border: 1px solid #cccccc;
                    border-radius: 8px;
                    padding: 5px;
                    padding-left: 15px;
                }
            """)

            self.web.hide()
            self.title_label.show()
            self.title_label.setFixedHeight(28)

    def isChecked(self):
        return self._checked

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit(self.name)
