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
from PySide6.QtCore import QTimer
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
    def __init__(self, stacked_widget, runs, tag_descriptions, active_run):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs
        self.active_run = active_run

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_row_layout = QHBoxLayout()
        top_row_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # top_row_layout.addWidget(logo_button, alignment=Qt.AlignmentFlag.AlignTop)

        # Spacer to push buttons to the right
        # top_row_layout.addStretch()

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
            btn.setFixedSize(40, 30)
            btn.setStyleSheet(
                """
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
            """
            )
            if tooltip:
                btn.setToolTip(tooltip)
            btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
            return btn

        # --- Sidebar links ---
        # button_layout.addWidget(make_link_button("📘", "https://github.com/mammoth-eu/mammoth-commons"))
        # button_layout.addWidget(make_link_button("📖", "https://github.com/mammoth-eu/FairnessDefinitionGuide"))
        button_layout.addWidget(
            make_link_button("🌐", "https://mammoth-eu.github.io/mammoth-commons/")
        )

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

        # --- Informational Sections ---

        info_container = QVBoxLayout()
        info_container.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_container.setSpacing(16)

        def make_info_box(html_content):
            frame = QFrame(self)
            frame.setObjectName("InfoBox")
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setStyleSheet(
                """
                        QFrame#InfoBox {
                            background-color: #dddddd;
                            border: 1px solid #e2e8f0;
                            border-radius: 10px;
                            padding: 14px 18px;
                        }
                        QLabel {
                            color: #334155;
                            font-size: 13px;
                            line-height: 1.4em;
                        }
                        a {
                            color: #0369a1;
                            text-decoration: none;
                            font-weight: 600;
                        }
                        a:hover {
                            text-decoration: underline;
                        }
                        ul {
                            margin-left: 16px;
                        }
                        li {
                            margin: 4px 0;
                        }
                    """
            )
            label = QLabel(html_content, frame)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setOpenExternalLinks(True)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(label)
            return frame

        # 1️⃣ Fairness is multi-layered
        fairness_html = """
                <p>Fairness is multi-layered in that it needs to account for various aspects, 
                such as technical, social, legal, and ethical. MAI-BIAS is meant for AI system creators, 
                so it focuses on the technical aspects. However, these make up only a part of the problem; 
                we recommend close cooperation with other disciplines to properly address the issue of fairness:.</p>

                💡 Consult with legal experts to ensure compliance with laws and regulations.
                <br>💡 Work with social scientists to gather interests of 
                <span title="Stakeholders refer to individuals or social groups who might be positively or negatively affected by the use of AI. They include, for example, developers, users, profiting organizations, policymakers, and vulnerable groups who might be discriminated against by its use. They may also include product owners, such as parent or funding organizations, that drive the system’s main technical specifications.">
                stakeholders</span> and ensure that they are adequately represented and integrated.
                <br>💡 Combine research principles with fairness concerns. This requires co-designing AI systems with said stakeholders.</li>
                <br><br>
                
                
                <a href='https://github.com/mammoth-eu/FairnessDefinitionGuide' target='_blank'>AI fairness definition guide</a><br/>
                <span>Learn more about an interdisciplinary approach to fairness in this guide by the MAMMOth project.</span>
                <br>
                <a href='https://www.trail-ml.com/eu-ai-act-compliance-checker' target='_blank'>Am I affected by the EU AI Act?</a><br/>
                <span>Visit this self-assessment checklist by the third-party European AI Alliance.</span>
                """
        info_container.addWidget(make_info_box(fairness_html))

        self.main_layout.addLayout(info_container)

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
        scroll_bar = self.scroll_area.verticalScrollBar()
        scroll_value = scroll_bar.value()

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
        card_width = 1100
        card_height = 40
        card_spacing = 6
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
        logo_card.setFixedSize(card_width, card_height * 3)
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
        img_max_height = int(card_height * 2)
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
            (card_height * 3 - logo_pixmap.height()) // 2,
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

            # --- Compact one-line layout instead of stacked sections ---
            card_layout = QHBoxLayout(card_widget)
            card_layout.setContentsMargins(10, 6, 10, 6)
            card_layout.setSpacing(8)

            # --- Title / status label ---
            desc_label = QLabel(
                (
                    get_special_title(latest_run)
                    if latest_run["status"] == "completed"
                    else "INCOMPLETE"
                ),
                card_widget,
            )
            desc_label.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {card_border}; border: none; background: none;"
            )
            desc_label.setFixedHeight(26)
            desc_label.setFixedWidth(360)
            card_layout.addWidget(desc_label)

            # --- Tags inline (dataset/model/analysis) ---
            tags_row = QHBoxLayout()
            tags_row.setSpacing(4)
            tags_row.setContentsMargins(0, 0, 0, 0)
            tags_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            for key in ["dataset", "model", "analysis"]:
                mod = latest_run.get(key, {}).get("module", "")
                if mod:
                    tag_btn = self.create_tag_button(
                        f"{mod}",
                        "Module info",
                        partial(lambda mod=mod: self.show_tag_description(mod)),
                    )
                    tag_btn.setFixedHeight(24)
                    tags_row.addWidget(tag_btn)
            tags_widget = QWidget(card_widget)
            tags_widget.setLayout(tags_row)

            # --- Timestamp ---
            timestamp_label = QLabel(
                convert_to_readable(latest_run["timestamp"]),
                # if latest_run["status"] == "completed"
                # else "not yet run",
                card_widget,
            )
            timestamp_label.setFixedWidth(140)
            timestamp_label.setStyleSheet(
                "font-size: 12px; color: #666; background: none; border: none;"
            )
            timestamp_label.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
            )
            card_layout.addWidget(timestamp_label)

            # --- Spacer ---
            card_layout.addWidget(tags_widget)
            card_layout.addStretch()

            # --- Actions inline (History, New, Delete) ---
            if len(runs) > 1 and len(latest_per_group) != 1:
                history_btn = QPushButton(f"History ({len(runs)})", card_widget)
                history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                history_btn.setFixedHeight(26)
                history_btn.setFixedWidth(100)
                history_btn.setStyleSheet(
                    """
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
                """
                )
                group_run_indices = [idx for idx, _ in runs]

                def make_on_history(indices):
                    def on_history():
                        self.invisible_runs = set(range(len(self.runs))) - set(indices)
                        self.refresh_dashboard()

                    return on_history

                history_btn.clicked.connect(make_on_history(group_run_indices))
                card_layout.addWidget(history_btn)

            if latest_run["status"] == "completed":
                card_layout.addWidget(
                    self.create_icon_button(
                        "+",
                        "#007bff",
                        "New variation",
                        partial(lambda i=latest_index: self.create_variation(i)),
                        size=26,
                    )
                )

            card_layout.addWidget(
                self.create_icon_button(
                    "🗑",
                    "#dc3545",
                    "Delete",
                    partial(lambda i=latest_index: self.delete_item(i)),
                    size=26,
                )
            )

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
            card_widget.mousePressEvent = partial(
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

        if (
            len(latest_per_group) <= 1  # and len(self.invisible_runs) > 0
        ):  # or (len(latest_per_group) == 1 and len(runs) > 1):
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
                narrow_card.setFixedSize(narrow_width, 35)
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
                narrow_layout.setContentsMargins(7, 3, 7, 3)
                narrow_layout.setSpacing(2)

                # --- Special title and date ---
                info_label = QLabel(
                    "<b>{}</b> <span style='color:#666'>{}</span>".format(
                        (
                            get_special_title(run)
                            if run["status"] == "completed"
                            else "INCOMPLETE"
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
                delete_button = self.create_icon_button(
                    "🗑",
                    "#dc3545",
                    "Delete",
                    partial(lambda i=index: self.delete_item(i)),
                    size=25,
                )
                narrow_layout.addWidget(
                    delete_button,
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

        def restore_scroll_position():
            sb = self.scroll_area.verticalScrollBar()
            sb.setValue(scroll_value)

        QTimer.singleShot(0, restore_scroll_position)

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
