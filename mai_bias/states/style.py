from PySide6.QtWidgets import QPushButton, QWidget, QSizePolicy


class Styled(QWidget):
    def create_icon_button(self, text, color, tooltip, callback, size=30):
        button = QPushButton(text, self)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color}; 
                color: white; 
                border-radius: 5px;
                font-size: {size*6//8 if text=='+' else size//2}px;
                border: 1px solid black;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border: 2px solid black;
                background-color: {self.highlight_color(color)};
            }}
        """
        )

        button.setFixedSize(size, size)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def create_tag_button(self, text, tooltip, callback):
        button = QPushButton(text, self)

        button.setStyleSheet(
            """
            QPushButton {
                color: #222;
                border-radius: 10px;
                font-size: 12px;
                border: 0px solid #bbb;
                padding: 0px 5px;
            }
            QPushButton:hover {
                border-radius: 10px;
                border: 2px solid #888;
                background-color: #f5f5f5;
            }
            QPushButton:pressed {
                background-color: #dddddd;
            }
            """
        )

        # button.setStyleSheet("""
        #     QPushButton {
        #         border-radius: 7px;
        #         background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e0f2fe, stop:1 #bae6fd);
        #         color: #0369a1;
        #         padding: 2px 10px;
        #         font-size: 13px;
        #         border: 1.2px solid #7dd3fc;
        #         font-weight: 500;
        #     }
        #     QPushButton:hover {
        #         background: #38bdf8;
        #         color: white;
        #         border: 1.5px solid #0ea5e9;
        #     }
        # """)
        button.setFixedHeight(20)

        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        # button.setFixedSize(180, 30)

        return button

    def highlight_color(self, color):
        if color.startswith("#"):
            color = color[1:]
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r = min(r + 22, 255)
        g = min(g + 22, 255)
        b = min(b + 22, 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def darker_color(self, color):
        if color.startswith("#"):
            color = color[1:]
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r = max(r - 64, 0)
        g = max(g - 64, 0)
        b = max(b - 64, 0)
        return f"#{r:02x}{g:02x}{b:02x}"
