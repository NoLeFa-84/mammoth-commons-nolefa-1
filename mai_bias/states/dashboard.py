from PySide6.QtWidgets import (
    QLabel,
    QGridLayout,
    QWidget,
    QHBoxLayout,
    QScrollArea,
    QMessageBox,
    QLineEdit,
    QDialog,
    QVBoxLayout,
    QPushButton,
)
from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt, QUrl
from datetime import datetime, timedelta
from mammoth_commons.externals import prepare, prepare_html, SEPARATOR
from PySide6.QtGui import QPixmap, QDesktopServices
from functools import partial
from PySide6.QtWebEngineWidgets import QWebEngineView
from .cache import ExternalLinkPage
from .step import save_all_runs
from .style import Styled
import re
from collections import defaultdict

EN_MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


def get_timestamp(run):
    return run.get("timestamp") or ""


def _category_from_ts(ts_str: str) -> str:
    try:
        dt = datetime.strptime(ts_str, "%y-%m-%d %H:%M")
    except Exception:
        return ""
    now = datetime.now()
    if now - dt < timedelta(hours=1):
        return "Last hour"
    if dt.date() == now.date():
        return "Today"
    if dt.date() == (now - timedelta(days=1)).date():
        return "Yesterday"
    if dt.year == now.year and dt.month == now.month:
        return "This month"
    return f"{EN_MONTHS[dt.month]} {dt.year}"


def convert_to_readable(date_str):
    dt = datetime.strptime(date_str, "%y-%m-%d %H:%M")
    return f"{dt.day} {EN_MONTHS[dt.month]} {dt.year} - {dt.strftime('%H:%M')}"


class Dashboard(Styled):
    def __init__(self, stacked_widget, runs, tag_descriptions, active_run):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs
        self.active_run = active_run

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_row_layout = QHBoxLayout()
        top_row_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

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
        button_layout.addWidget(
            self.new_action(
                "🌐",
                "#0369a1",
                "Developer portal",
                lambda: QDesktopServices.openUrl(
                    QUrl("https://mammoth-eu.github.io/mammoth-commons/")
                ),
            )
        )
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        top_row_layout.addWidget(button_widget, alignment=Qt.AlignmentFlag.AlignTop)
        self.main_layout.addLayout(top_row_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setStyleSheet("""
            QScrollArea {border: none; background: transparent;}
            QScrollArea QWidget {background: transparent;}
            QScrollBar:vertical, QScrollBar:horizontal {border: none;background: transparent;}
            """)

        # --- LOGO CARD ---
        logo_card = QPushButton(self)
        logo_card.setText("\n\n\n\n\n\nStart new MAI-BIAS analysis")
        logo_card.setCursor(Qt.CursorShape.PointingHandCursor)
        logo_card.clicked.connect(self.create_new_item)
        logo_card.setStyleSheet(f"""
            QPushButton {{background-color: white; border-left: 3px solid #0369a1; border-radius: 0px; padding: 0px;}}
            QPushButton:hover {{background-color: #d3ecfa; border-left: 4px solid #0369a1;}}
            """)
        logo_pixmap = QPixmap(prepare("https:///icons/mai_bias.png"))
        # Fit logo to ~60% width of card, keep aspect
        img_max_width = int(1100 * 0.60)
        img_max_height = int(40 * 2)
        logo_pixmap = logo_pixmap.scaled(
            img_max_width,
            img_max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_label = QLabel(logo_card)
        logo_label.setPixmap(logo_pixmap)
        self.logo_pixmap = logo_pixmap
        self.logo_card = logo_card
        self.logo_label = logo_label
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter
        )
        self.layout.setSpacing(0)
        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

        info_container = QVBoxLayout()
        info_container.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_container.setSpacing(16)
        self.main_layout.addLayout(info_container)
        self.setLayout(self.main_layout)
        self.tag_descriptions = tag_descriptions
        self.hidden = set()
        self.refresh_dashboard()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_dashboard()

    def filter_runs(self, text):
        if not self.runs:
            return
        prev = self.hidden
        self.hidden = set()
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
            self.hidden.add(index)
        # refresh but only if something changed
        if len(prev - self.hidden) == 0 and len(self.hidden - prev) == 0:
            return
        self.refresh_dashboard()

    def view_result(self, index):
        self.active_run[-1] = self.runs[index]
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
        self.active_run[-1] = self.runs[index]
        self.stacked_widget.slideToWidget(1)

    def create_variation(self, index):
        new_run = self.runs[index].copy()
        new_run["status"] = "new"
        new_run["timestamp"] = now()
        self.active_run[-1] = new_run
        self.runs.append(new_run)
        self.stacked_widget.slideToWidget(1)

    def create_new_item(self):
        new_run = {"description": "", "timestamp": now(), "status": "in_progress"}
        self.active_run[-1] = new_run
        self.runs.append(new_run)
        self.stacked_widget.slideToWidget(1)
        self.refresh_dashboard()

    def delete_item(self, index, confirm=True):
        if (
            confirm
            and QMessageBox.question(
                self,
                "Delete?",
                f"The analysis, which is the most recent of its kind,\nwill be permanently deleted. This message does\nnot appear for older history items.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.runs.pop(index)
        self.notify_delete(index)
        self.refresh_dashboard()
        save_all_runs("history.json", self.runs)

    def notify_delete(self, index):
        self.hidden = {i - 1 if i > index else i for i in self.hidden if i != index}

    def clear_layout(self, layout):
        if not layout:
            return
        while layout.count():
            child = layout.takeAt(0)
            widget = child.widget()
            if widget:
                if (
                    widget != self.logo_card
                    and widget != self.logo_pixmap
                    and widget != self.logo_label
                ):
                    widget.deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())

    def showEvent(self, event):
        self.refresh_dashboard()

    def refresh_dashboard(self):
        scroll_bar = self.scroll_area.verticalScrollBar()
        scroll_value = scroll_bar.value()
        self.clear_layout(self.layout)
        groups = defaultdict(list)
        for i, run in enumerate(self.runs):
            if i in self.hidden:
                continue
            group_key = (
                run["description"],
                run.get("dataset", {}).get("module", ""),
                run.get("model", {}).get("module", ""),
                run.get("analysis", {}).get("module", ""),
            )
            groups[group_key].append((i, run))

        latest_per_group = {}
        for group_key, runs in groups.items():
            runs_sorted = sorted(runs, key=lambda x: get_timestamp(x[1]), reverse=True)
            latest_per_group[group_key] = runs_sorted

        # --- Card layout constants ---
        card_width = (self.width() or 1200) - 80
        card_height = 50
        card_spacing = 6
        max_cols = 1

        current_category = None
        grid_layout = QGridLayout()
        grid_layout.setSpacing(card_spacing)
        row = 0
        col = 0

        logo_card = self.logo_card
        logo_label = self.logo_label
        logo_pixmap = self.logo_pixmap
        logo_card.setFixedSize(card_width, card_height * 3)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setGeometry(
            (card_width - logo_pixmap.width()) // 2,
            (card_height * 3 - logo_pixmap.height()) // 2,
            logo_pixmap.width(),
            logo_pixmap.height(),
        )
        if not self.hidden:
            logo_card.show()
            grid_layout.addWidget(logo_card, row, col)
            col += 1
        else:
            logo_card.hide()
        if col >= max_cols:
            row += 1
            col = 0

        def _maybe_add_separator(cat: str):
            """Insert a full‑width QLabel if *cat* is non‑empty and different
            from the previously printed heading."""
            nonlocal row, col, current_category
            if not cat or cat == current_category:
                return
            sep_lbl = QLabel(cat, self)
            sep_lbl.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    color: #444;
                    background: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                """)
            grid_layout.addWidget(sep_lbl, row, 0, 1, max_cols)  # span all columns
            row += 1
            col = 0
            current_category = cat

        # --- RESULT CARDS ---
        for group_key, runs in sorted(
            latest_per_group.items(),
            key=lambda kv: get_timestamp(
                kv[1][0][1]
            ),  # timestamp of the first (newest) run
            reverse=True,
        ):
            latest_index, latest_run = runs[0]
            _maybe_add_separator(_category_from_ts(latest_run.get("timestamp", "")))
            card_widget = QWidget(self)
            card_widget.setCursor(Qt.CursorShape.PointingHandCursor)
            card_widget.setObjectName("ResultCard")
            card_widget.setFixedSize(card_width, card_height)
            special = get_special_title(latest_run).lower()
            if "fail" in special or "bias" in special:
                card_border = "#b91c1c"
            elif any(
                word in special
                for word in ["report", "audit", "scan", "analysis", "explanation"]
            ):
                card_border = "#0369a1"
            else:
                card_border = "#047857"
            if latest_run["status"] != "completed":
                card_border = "#ca8a04"

            card_widget.setStyleSheet(f"""
                QWidget#ResultCard {{
                    background: white;
                    border-left: 3px solid {card_border};
                    border-radius: 0px;
                }}
                QWidget#ResultCard:hover {{
                    background: #EEEEEE;
                    border-left: 4px solid {card_border};
                    border-radius: 0px;
                }}
            """)
            card_layout = QHBoxLayout(card_widget)
            card_layout.setContentsMargins(10, 6, 10, 6)
            card_layout.setSpacing(2)

            # --- Title / status label ---
            desc_label = QLabel(
                (
                    "<b>"
                    + latest_run.get("analysis", {}).get("module", "")[0]
                    + latest_run.get("analysis", {}).get("module", "")[1:].lower()
                    + " for "
                    + latest_run.get("dataset", {}).get("module", "").lower()
                    + " and "
                    + latest_run.get("model", {}).get("module", "").lower()
                    + " model </b><br>"
                    + get_special_title(latest_run)
                    if latest_run["status"] == "completed"
                    else "INCOMPLETE"
                ).replace("model model", "model"),
                card_widget,
            )
            desc_label.setStyleSheet(
                f"font-size: 13px; color: black; border: none; background: none;"
            )
            desc_label.setFixedHeight(40)
            card_layout.addWidget(desc_label)
            timestamp_label = QLabel(
                convert_to_readable(latest_run["timestamp"]),
                card_widget,
            )
            timestamp_label.setFixedWidth(160)
            timestamp_label.setStyleSheet(
                "font-size: 12px; color: #666; background: none; border: none;padding-right:2px"
            )
            timestamp_label.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
            )
            card_layout.addStretch()
            card_layout.addWidget(timestamp_label)

            # --- Actions inline (History, New, Delete) ---
            if len(runs) > 1 and len(latest_per_group) != 1:
                history_btn = QPushButton(f"History ({len(runs)})", card_widget)
                history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                history_btn.setFixedHeight(26)
                history_btn.setFixedWidth(100)
                history_btn.setStyleSheet("""
                    QPushButton {
                        background: #f1f5f9;
                        border-radius: 5px;
                        border: 1px solid #b6c6da;
                        color: #0369a1;
                        font-size: 12px;
                        font-weight: 500;
                        padding: 0 10px;
                    }
                    QPushButton:hover {
                        background: #bae6fd;
                        color: #035388;
                        border: 1px solid #38bdf8;
                    }
                """)
                group_run_indices = [idx for idx, _ in runs]

                def make_on_history(indices):
                    def on_history():
                        self.hidden = set(range(len(self.runs))) - set(indices)
                        self.refresh_dashboard()

                    return on_history

                history_btn.clicked.connect(make_on_history(group_run_indices))
                card_layout.addWidget(history_btn)

            if latest_run["status"] == "completed":
                card_layout.addWidget(
                    self.new_action(
                        "+",
                        "#FFFFFF",
                        "New variation",
                        partial(lambda i=latest_index: self.create_variation(i)),
                        size=26,
                    )
                )

            card_layout.addWidget(
                self.new_action(
                    "X",
                    "#FFFFFF",
                    "Delete",
                    partial(lambda i=latest_index: self.delete_item(i)),
                    size=26,
                )
            )

            def card_mouse_press(
                event,
                i=latest_index,
                r=latest_run,
                runs_in_group=(idx for idx, _ in runs),
            ):
                pos = (
                    event.position().toPoint()
                    if hasattr(event, "position")
                    else event.pos()
                )
                for btn in card_widget.findChildren(QPushButton):
                    local_pos = btn.mapFromParent(pos)
                    if btn.rect().contains(local_pos):
                        return
                if r["status"] == "completed":
                    self.view_result(i)
                else:
                    self.edit_item(i)

            card_widget.mousePressEvent = partial(
                lambda event, i=latest_index, r=latest_run, runs_in_group=(
                    idx for idx, _ in runs
                ): card_mouse_press(event, i, r, runs_in_group)
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
            no_results_label.setStyleSheet("""
                color: #666;
                font-size: 15px;
                padding: 20px;
            """)
            # Add to a full-width row under the logo card (use next grid row, col=0 spanning all columns)
            grid_layout.addWidget(no_results_label, row, 0, 1, max_cols)
            row += 1

        if (
            len(latest_per_group) <= 1  # and len(self.hidden) > 0
        ):  # or (len(latest_per_group) == 1 and len(runs) > 1):
            # --- Clear Search Button ---
            clear_search_btn = QPushButton("Back", self)
            clear_search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            clear_search_btn.setStyleSheet("""
                QPushButton {
                    background: #7c2d12;
                    border-radius: 0px;
                    border-left: 3px solid #ea580c; /* Strong orange border */
                    color: #fde68a;
                    font-size: 13px;
                    font-weight: 500;
                    padding: 6px 22px;
                }
                QPushButton:hover {
                    background: #a53f13;         /* Brighter/darker orange on hover */
                    color: #fff7ed;              /* Lighter text on hover */
                    border-left: 4px solid #fb923c; /* Lighter orange border on hover */
                }
            """)

            def on_clear_search():
                self.search_field.setText("")
                self.hidden = set()
                self.refresh_dashboard()

            clear_search_btn.clicked.connect(on_clear_search)
            grid_layout.addWidget(clear_search_btn, row, 0, 1, max_cols)
            row += 1

        if len(latest_per_group) == 1 and len(runs) > 1:
            for sub_index, (index, run) in enumerate(
                sorted(runs[1:], key=lambda x: get_timestamp(x[1]), reverse=True)
            ):
                _maybe_add_separator(_category_from_ts(run.get("timestamp", "")))
                special = get_special_title(run).lower()
                if "fail" in special or "bias" in special:
                    narrow_border = "#b91c1c"
                elif any(
                    word in special
                    for word in ["report", "audit", "scan", "analysis", "explanation"]
                ):
                    narrow_border = "#0369a1"
                else:
                    narrow_border = "#047857"
                if run["status"] != "completed":
                    narrow_border = "#ca8a04"

                narrow_card = QWidget(self)
                narrow_card.setObjectName("NarrowResultCard")
                narrow_card.setCursor(Qt.CursorShape.PointingHandCursor)
                narrow_card.setFixedSize(card_width, 35)
                narrow_card.setStyleSheet(f"""
                    QWidget#NarrowResultCard {{
                        background: white;
                        border-left: 3px solid {narrow_border};
                        border-radius: 0px;
                    }}
                    QWidget#NarrowResultCard:hover {{
                        border-left: 4px solid {narrow_border};
                        background: #EEEEEE;
                    }}
                """)
                h_layout = QHBoxLayout(narrow_card)
                h_layout.setContentsMargins(7, 3, 7, 3)
                h_layout.setSpacing(2)
                title_txt = (
                    get_special_title(run)
                    if run["status"] == "completed"
                    else "INCOMPLETE"
                )
                title_lbl = QLabel(title_txt, narrow_card)
                title_lbl.setStyleSheet(
                    "border:none;background:none;font-size:12px;color:#000;"
                )
                h_layout.addWidget(title_lbl)
                h_layout.addStretch()
                ts_lbl = QLabel(convert_to_readable(run["timestamp"]), narrow_card)
                ts_lbl.setStyleSheet(
                    "border:none;background:none;font-size:12px;color:#666;padding-left:8px;"
                )
                ts_lbl.setAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
                )
                h_layout.addWidget(ts_lbl)
                delete_button = self.new_action(
                    "X",
                    "#FFFFFF",
                    "Delete",
                    partial(lambda i=index: self.delete_item(i, confirm=False)),
                    size=24,
                )
                h_layout.addWidget(delete_button)

                def narrow_card_mouse_press(event, i=index, r=run):
                    if event.button() == Qt.MouseButton.LeftButton:
                        pos = (
                            event.position()
                            if hasattr(event, "position")
                            else event.pos()
                        )
                        # ignore clicks on any child button
                        for b in narrow_card.findChildren(QPushButton):
                            if b.geometry().contains(int(pos.x()), int(pos.y())):
                                return
                        if r["status"] == "completed":
                            self.view_result(i)
                        else:
                            self.edit_item(i)

                narrow_card.mousePressEvent = narrow_card_mouse_press
                grid_layout.addWidget(narrow_card, row, col)
                col += 1
                if col >= max_cols:
                    row += 1
                    col = 0

        def restore_scroll_position():
            sb = self.scroll_area.verticalScrollBar()
            sb.setValue(scroll_value)

        QTimer.singleShot(0, restore_scroll_position)

    def show_tag_description(self, tag):
        dialog = QDialog(self)
        dialog.setWindowTitle("Module info")
        dialog.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(dialog)

        browser = QWebEngineView(dialog)
        browser.setFixedHeight(800)
        browser.setFixedWidth(800)

        html = self.tag_descriptions.get(tag, "No description available.")
        html = f"""
        <html>
        <head>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{font-family: Arial, sans-serif;font-size: 14px;color: #333; background-color: white; padding: 10px;}}
                h1   {{font-size: 18px; color: #0055aa;}}
                img  {{max-width: 100%; border: 1px solid #ccc; border-radius: 4px;}}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        browser.setPage(ExternalLinkPage(browser))
        browser.setHtml(prepare_html(html), QUrl("file:///"))
        layout.addWidget(browser)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        dialog.exec()


def get_special_title(run):
    ret = ""
    try:
        match = re.search(
            r"<h1\b[^>]*>.*?</h1>",
            run.get("analysis", dict()).get("return", ""),
            re.DOTALL,
        )
        if match:
            ret = match.group().replace("h1", "span")
    except Exception:
        pass
    if run.get("analysis", {}).get("params", {}).get("sensitive", ""):
        ret += (
            SEPARATOR
            + "Sensitive: <i>"
            + run.get("analysis", {}).get("params", {}).get("sensitive", "")
            + "</i>"
        )
    return ret
