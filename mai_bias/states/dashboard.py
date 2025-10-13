from PySide6.QtWidgets import (
    QLabel,
    QGridLayout,
    QWidget,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QMessageBox,
    QLineEdit,
    QDialog,
    QVBoxLayout,
    QPushButton,
)
from PySide6.QtCore import Qt, QUrl
from datetime import datetime
from mammoth_commons.externals import prepare
from PySide6.QtGui import QPixmap, QDesktopServices
from functools import partial
from PySide6.QtWebEngineWidgets import QWebEngineView
from .cache import ExternalLinkPage
from .step import save_all_runs
from .style import Styled
import re


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


ENGLISH_MONTHS = [
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def convert_to_readable(date_str):
    dt = datetime.strptime(date_str, "%y-%m-%d %H:%M")
    return f"{dt.day} {ENGLISH_MONTHS[dt.month]} {dt.year} - {dt.strftime('%H:%M')}"


class Dashboard(Styled):
    def __init__(self, stacked_widget, runs, tag_descriptions):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_row_layout = QHBoxLayout()
        top_row_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # top_row_layout.addWidget(logo_button, alignment=Qt.AlignmentFlag.AlignTop)

        # Spacer to push buttons to the right
        #top_row_layout.addStretch()

        # Buttons on the right
        search_field = QLineEdit(self)
        search_field.setPlaceholderText("Search for title or module...")
        search_field.setFixedSize(200, 30)
        search_field.textChanged.connect(self.filter_runs)
        self.search_field = search_field

        button_layout = QHBoxLayout()
        button_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter
        )
        button_layout.addWidget(search_field)


        def make_link_button(text, url, tooltip=None):
            btn = QPushButton(text, self)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedSize(
            40, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background: white;
                    color: #0369a1;
                    font-weight: 600;
                    border: 0px solid #0369a1;
                    border-radius: 8px;
                    text-align: center;
                    font-size:20px;
                }
                QPushButton:hover {
                    background: #d3ecfa;
                }
            """)
            if tooltip:
                btn.setToolTip(tooltip)
            btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
            return btn
        # --- Sidebar links ---
        #button_layout.addWidget(make_link_button("📘", "https://github.com/mammoth-eu/mammoth-commons"))
        #button_layout.addWidget(make_link_button("📖", "https://github.com/mammoth-eu/FairnessDefinitionGuide"))
        button_layout.addWidget(make_link_button("🌐", "https://mammoth-eu.github.io/mammoth-commons/"))

        # Wrap buttons in a widget so layout behaves properly
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        top_row_layout.addWidget(button_widget, alignment=Qt.AlignmentFlag.AlignTop)

        # Add everything to the main layout
        self.main_layout.addLayout(top_row_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea QWidget {
                background: transparent;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                border: none;
                background: transparent;
            }
        """
        )

        # Content Widget
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter
        )
        self.layout.setSpacing(0)
        self.scroll_area.setWidget(self.content_widget)

        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)
        self.tag_descriptions = tag_descriptions

        self.invisible_runs = set()
        self.refresh_dashboard()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_dashboard()

    def filter_runs(self, text):
        if not self.runs:
            return
        prev = self.invisible_runs
        self.invisible_runs = set()
        for index, run in enumerate(self.runs):
            fields = [
                run["description"].lower(),
                run.get("dataset", dict()).get("module", "").lower(),
                run.get("model", dict()).get("module", "").lower(),
                run.get("analysis", dict()).get("module", "").lower(),
                get_special_title(run).lower(),
            ]
            if any(text.lower() in field for field in fields):
                continue
            self.invisible_runs.add(index)
        # refresh but only if something changed
        if (
            len(prev - self.invisible_runs) == 0
            and len(self.invisible_runs - prev) == 0
        ):
            return
        self.refresh_dashboard()

    def view_result(self, index):
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.slideToWidget(4)

    def edit_item(self, index):
        if self.runs[index].get("status", "") != "completed":
            reply = QMessageBox.StandardButton.Yes
        else:
            reply = QMessageBox.question(
                self,
                "Edit?",
                f"You can change modules and modify parameters. "
                "However, this will also remove its results. Consider creating a variation if you want to preserve current results.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # self.runs[index]["timestamp"] = now()
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.slideToWidget(1)

    def create_variation(self, index):
        new_run = self.runs[index].copy()
        new_run["status"] = "new"
        new_run["timestamp"] = now()
        self.runs.append(new_run)
        self.stacked_widget.slideToWidget(1)

    def create_new_item(self):
        self.runs.append(
            {"description": "", "timestamp": now(), "status": "in_progress"}
        )
        self.stacked_widget.slideToWidget(1)
        self.refresh_dashboard()

    def delete_item(self, index):
        reply = QMessageBox.question(
            self,
            "Delete?",
            f"The analysis will be permanently deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.runs.pop(index)
        self.refresh_dashboard()
        save_all_runs("history.json", self.runs)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def showEvent(self, event):
        self.refresh_dashboard()

    def refresh_dashboard(self):
        self.clear_layout(self.layout)
        from collections import defaultdict

        groups = defaultdict(list)
        for i, run in enumerate(self.runs):
            if i in self.invisible_runs:
                continue
            group_key = (
                run["description"],
                run.get("dataset", {}).get("module", ""),
                run.get("model", {}).get("module", ""),
                run.get("analysis", {}).get("module", ""),
            )
            groups[group_key].append((i, run))

        def get_timestamp(run):
            return run.get("timestamp") or ""

        latest_per_group = {}
        for group_key, runs in groups.items():
            runs_sorted = sorted(runs, key=lambda x: get_timestamp(x[1]), reverse=True)
            latest_per_group[group_key] = runs_sorted

        # --- Card layout constants ---
        card_width = 480
        card_height = 110
        card_spacing = 9
        # Responsive cols
        window_width = self.scroll_area.viewport().width() or 700
        max_cols = max(1, window_width // (card_width + card_spacing))
        if len(latest_per_group) == 1:
            max_cols = 1


        grid_layout = QGridLayout()
        grid_layout.setSpacing(card_spacing)
        row = 0
        col = 0

        # --- LOGO CARD ---
        logo_card = QPushButton(self)
        logo_card.setCursor(Qt.CursorShape.PointingHandCursor)
        logo_card.setFixedSize(card_width, card_height)
        logo_card.setToolTip("New analysis")
        logo_card.clicked.connect(self.create_new_item)
        logo_card.setStyleSheet(
            f"""
            QPushButton {{
                background-color: white;
                border: 2px dashed #0369a1;
                border-radius: 10px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: #d3ecfa;
                border: 2px solid #0369a1;
            }}
        """
        )

        # Centered logo image
        logo_pixmap = QPixmap(
            prepare(
                "https://raw.githubusercontent.com/mammoth-eu/mammoth-commons/dev/mai_bias/logo.png"
            )
        )
        # Fit logo to ~60% width of card, keep aspect
        img_max_width = int(card_width * 0.60)
        img_max_height = int(card_height * 0.7)
        logo_pixmap = logo_pixmap.scaled(
            img_max_width,
            img_max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        logo_label = QLabel(logo_card)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setGeometry(
            (card_width - logo_pixmap.width()) // 2,
            (card_height - logo_pixmap.height()) // 2,
            logo_pixmap.width(),
            logo_pixmap.height(),
        )
        if not self.invisible_runs:
            grid_layout.addWidget(logo_card, row, col)
            col += 1
        if col >= max_cols:
            row += 1
            col = 0

        # --- RESULT CARDS ---
        for group_key, runs in latest_per_group.items():
            latest_index, latest_run = runs[0]

            card_widget = QWidget(self)
            card_widget.setObjectName("ResultCard")
            card_widget.setFixedSize(card_width, card_height)
            special = get_special_title(latest_run).lower()
            if "fail" in special or "bias" in special:
                card_border = "#b91c1c"  # deep red
                card_hover = "#fcd8dd"  # matte red
            elif any(
                word in special
                for word in ["report", "audit", "scan", "analysis", "explanation"]
            ):
                card_border = "#0369a1"  # deep blue
                card_hover = "#d3ecfa"  # matte blue
            else:
                card_border = "#047857"  # deep green
                card_hover = "#bff2c1"  # matte green (more green)
            if latest_run["status"] != "completed":
                card_border = "#ca8a04"  # deep yellow
                card_hover = "#fff7c2"  # matte yellow

            card_widget.setStyleSheet(
                f"""
                QWidget#ResultCard {{
                    background: white;
                    border: 1px solid {card_border};
                    border-radius: 10px;
                }}
                QWidget#ResultCard:hover {{
                    background: {card_hover};
                    border: 2px solid {card_border};
                }}
            """
            )

            card_layout = QVBoxLayout(card_widget)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setSpacing(0)

            # --- Header Bar with title ---
            header_bar = QFrame(card_widget)
            header_bar.setFixedHeight(34)
            header_bar.setStyleSheet(
                f"""
                QFrame {{
                    background: {card_border};
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                }}
            """
            )
            header_layout = QHBoxLayout(header_bar)
            header_layout.setContentsMargins(11, 0, 11, 0)
            header_layout.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            desc_label = QLabel(
                (
                    get_special_title(latest_run)
                    if latest_run["status"] == "completed"
                    else "INCOMPLETE"
                ),
                header_bar,
            )
            desc_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: white; border: none; background: none;"
            )
            header_layout.addWidget(desc_label)
            header_layout.addStretch()
            card_layout.addWidget(header_bar)

            # --- Separator ---
            separator = QFrame(card_widget)
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            separator.setStyleSheet(
                "color: #e5e7eb; background: #e5e7eb; min-height: 1px; max-height: 1px; border: none;"
            )
            card_layout.addWidget(separator)

            # --- Tags---
            tags_col = QHBoxLayout()
            tags_col.setContentsMargins(11, 5, 0, 5)
            tags_col.setSpacing(2)

            for key in ["dataset", "model", "analysis"]:
                mod = latest_run.get(key, {}).get("module", "")
                if mod:
                    tag_btn = self.create_tag_button(
                        f" {mod} ",
                        "Module info",
                        partial(lambda mod=mod: self.show_tag_description(mod)),
                    )
                    tags_col.addWidget(tag_btn)

            tags_col.addStretch()
            card_layout.addLayout(tags_col)

            # --- Main Content (info and actions) ---
            main_content = QWidget(card_widget)
            main_layout = QVBoxLayout(main_content)
            main_layout.setContentsMargins(11, 5, 11, 0)
            main_layout.setSpacing(0)

            # Actions row
            actions_row = QHBoxLayout()
            actions_row.setContentsMargins(0, 0, 0, 5)
            actions_row.setSpacing(2)

            info_label = QLabel(
                "<span style='font-size:12px;color:#666'>{}</span>".format(
                    convert_to_readable(latest_run["timestamp"])
                    if latest_run["status"] == "completed"
                    else ""
                ),
                main_content,
            )
            info_label.setTextFormat(Qt.TextFormat.RichText)
            info_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
            )
            info_label.setStyleSheet(
                "border: none; background: none; font-size: 12px; margin-top: 2px;"
            )
            actions_row.addWidget(info_label)

            actions_row.addStretch()
            if len(runs) > 1 and len(latest_per_group) != 1:
                history_btn = QPushButton("History (" + str(len(runs)) + ")", self)
                history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                history_btn.setFixedHeight(28)
                history_btn.setStyleSheet(
                    """
                    QPushButton {
                        background: #f1f5f9;
                        border-radius: 6px;
                        border: 1.1px solid #b6c6da;
                        color: #0369a1;
                        font-size: 13px;
                        font-weight: 500;
                        padding: 0 15px;
                    }
                    QPushButton:hover {
                        background: #bae6fd;
                        color: #035388;
                        border: 1.4px solid #38bdf8;
                    }
                """
                )
                group_run_indices = [idx for idx, _ in runs]

                def make_on_history(indices):
                    def on_history():
                        self.invisible_runs = set(range(len(self.runs))) - set(indices)
                        self.refresh_dashboard()

                    return on_history

                history_btn.clicked.connect(make_on_history(group_run_indices))
                actions_row.addWidget(history_btn)

            if latest_run["status"] == "completed":
                actions_row.addWidget(
                    self.create_icon_button(
                        "+",
                        "#007bff",
                        "New variation",
                        partial(lambda i=latest_index: self.create_variation(i)),
                        size=28,
                    )
                )
            actions_row.addWidget(
                self.create_icon_button(
                    "🗑",
                    "#dc3545",
                    "Delete",
                    partial(lambda i=latest_index: self.delete_item(i)),
                    size=28,
                )
            )
            main_layout.addLayout(actions_row)
            card_layout.addWidget(main_content)

            # --- Make card clickable except buttons and tags ---
            def card_mouse_press(
                event,
                i=latest_index,
                r=latest_run,
                runs_in_group=[idx for idx, _ in runs],
            ):
                # Get click pos as QPoint (ints)
                if hasattr(event, "position"):
                    pos = event.position().toPoint()
                else:
                    pos = event.pos()

                # Check if click was on a child button
                for btn in card_widget.findChildren(QPushButton):
                    local_pos = btn.mapFromParent(pos)
                    if btn.rect().contains(local_pos):
                        return
                if r["status"] == "completed":
                    self.view_result(i)
                else:
                    self.edit_item(i)

            # Assign directly; do NOT use lambda+partial, just a closure:
            card_widget.mousePressEvent = (
                lambda event, i=latest_index, r=latest_run, runs_in_group=[
                    idx for idx, _ in runs
                ]: card_mouse_press(event, i, r, runs_in_group)
            )

            grid_layout.addWidget(card_widget, row, col)
            col += 1
            if col >= max_cols:
                row += 1
                col = 0

        self.layout.addLayout(grid_layout)
        self.content_widget.adjustSize()

        if not latest_per_group and self.runs:
            no_results_label = QLabel("No results found.", self)
            no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_results_label.setStyleSheet(
                """
                color: #666;
                font-size: 15px;
                padding: 20px;
            """
            )
            # Add to a full-width row under the logo card (use next grid row, col=0 spanning all columns)
            grid_layout.addWidget(no_results_label, row, 0, 1, max_cols)
            row += 1

        if len(latest_per_group) <= 1:
            # --- Clear Search Button ---
            clear_search_btn = QPushButton("Back", self)
            clear_search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            clear_search_btn.setStyleSheet(
                """
                QPushButton {
                    background: #7c2d12;         /* Very dark orange background */
                    border-radius: 7px;
                    border: 1.1px solid #ea580c; /* Strong orange border */
                    color: #fde68a;              /* Light orange text for contrast */
                    font-size: 13px;
                    font-weight: 500;
                    padding: 6px 22px;
                }
                QPushButton:hover {
                    background: #a53f13;         /* Brighter/darker orange on hover */
                    color: #fff7ed;              /* Lighter text on hover */
                    border: 1.4px solid #fb923c; /* Lighter orange border on hover */
                }
            """
            )

            def on_clear_search():
                self.search_field.setText("")
                self.invisible_runs = set()
                self.refresh_dashboard()

            clear_search_btn.clicked.connect(on_clear_search)
            grid_layout.addWidget(clear_search_btn, row, 0, 1, max_cols)
            row += 1

        # if len(latest_per_group) == 1 and len(runs)==1:
        #     no_results_label = QLabel("Showing history." if  len(runs)>1 else "Found one run: no history.", self)
        #     no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #     no_results_label.setStyleSheet("""
        #         color: #666;
        #         font-size: 15px;
        #         padding: 20px;
        #     """)
        #     # Add to a full-width row under the logo card (use next grid row, col=0 spanning all columns)
        #     grid_layout.addWidget(no_results_label, row, 0, 1, max_cols)
        #     row += 1

        if len(latest_per_group) == 1 and len(runs) > 1:
            # other runs, sorted by timestamp DESC (latest first, skip runs[0])
            for sub_index, (index, run) in enumerate(
                sorted(runs[1:], key=lambda x: get_timestamp(x[1]), reverse=True)
            ):
                special = get_special_title(run).lower()
                if "fail" in special or "bias" in special:
                    narrow_border = "#b91c1c"  # deep red
                    narrow_bg = "#fcd8dd"  # matte red
                elif any(
                    word in special
                    for word in ["report", "audit", "scan", "analysis", "explanation"]
                ):
                    narrow_border = "#0369a1"  # deep blue
                    narrow_bg = "#d3ecfa"  # matte blue
                else:
                    narrow_border = "#047857"  # deep green
                    narrow_bg = "#bff2c1"  # matte green
                if run["status"] != "completed":
                    narrow_border = "#ca8a04"  # deep yellow
                    narrow_bg = "#fff7c2"  # matte yellow

                narrow_card = QWidget(self)
                narrow_card.setObjectName("NarrowResultCard")
                narrow_width = int(card_width)
                narrow_card.setFixedSize(narrow_width, 50)
                narrow_card.setStyleSheet(
                    f"""
                    QWidget#NarrowResultCard {{
                        background: {narrow_bg};
                        border: 1.8px solid {narrow_border};
                        border-radius: 7px;
                    }}
                    QWidget#NarrowResultCard:hover {{
                        border: 2.2px solid {narrow_border};
                        background: {self.highlight_color(narrow_bg)};
                    }}
                """
                )
                narrow_layout = QGridLayout(narrow_card)
                narrow_layout.setContentsMargins(7, 5, 7, 5)
                narrow_layout.setSpacing(2)

                # --- Special title and date ---
                info_label = QLabel(
                    "<b>{}</b><br><span style='font-size:11px;color:#666'>{}</span>".format(
                        (
                            get_special_title(run)
                            if run["status"] == "completed"
                            else "Creating"
                        ),
                        convert_to_readable(run["timestamp"]),
                    ),
                    self,
                )
                info_label.setTextFormat(Qt.TextFormat.RichText)
                info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                info_label.setStyleSheet(
                    "border: none; background: none; font-size: 12px; margin-top: 2px;"
                )
                narrow_layout.addWidget(info_label, 0, 0)
                narrow_layout.addWidget(
                    self.create_icon_button(
                        "🗑",
                        "#dc3545",
                        "Delete",
                        partial(lambda i=index: self.delete_item(i)),
                        size=28,
                    ),
                    0,
                    1,
                )

                def narrow_card_mouse_press(event, i=index, r=run):
                    if event.button() == Qt.MouseButton.LeftButton:
                        pos = (
                            event.position()
                            if hasattr(event, "position")
                            else event.pos()
                        )
                        for b in narrow_card.findChildren(QPushButton):
                            if b.geometry().contains(int(pos.x()), int(pos.y())):
                                return
                        if r["status"] == "completed":
                            self.view_result(i)
                        else:
                            self.edit_item(i)

                narrow_card.mousePressEvent = narrow_card_mouse_press

                # Add to grid (use next col/row, just like normal cards)
                grid_layout.addWidget(narrow_card, row, col)
                col += 1
                if col >= max_cols:
                    row += 1
                    col = 0

    def show_tag_description(self, tag):
        dialog = QDialog(self)
        dialog.setWindowTitle("Module info")
        layout = QVBoxLayout(dialog)

        browser = QWebEngineView(dialog)
        browser.setFixedHeight(300)
        browser.setFixedWidth(800)

        html = self.tag_descriptions.get(tag, "No description available.")
        html = f"""
        <html>
        <head>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 14px;
                color: #333;
                background-color: #fafafa;
                padding: 10px;
            }}
            h1 {{
                font-size: 18px;
                color: #0055aa;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
        </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        browser.setPage(ExternalLinkPage(browser))
        browser.setHtml(html)
        layout.addWidget(browser)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        dialog.exec()


def get_special_title(run):
    try:
        match = re.search(
            r"<h1\b[^>]*>.*?</h1>",
            run.get("analysis", dict()).get("return", ""),
            re.DOTALL,
        )
        if match:
            return match.group().replace("h1", "span")
    except Exception:
        pass
    return ""
