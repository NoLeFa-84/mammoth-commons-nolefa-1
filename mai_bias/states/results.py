from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt, QTimer
from .step import save_all_runs
from .style import Styled
from datetime import datetime
from PySide6.QtWebEngineWidgets import QWebEngineView
from .cache import ExternalLinkPage


def format_run(run):
    return run["description"] + " " + run["timestamp"]


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


class Results(Styled):
    def __init__(self, stacked_widget, runs, tag_descriptions, dataset):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs
        self.tag_descriptions = tag_descriptions
        self.dataset = dataset

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Top Row (Title & Buttons)
        self.top_container = QHBoxLayout()
        self.top_container.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Title label (Now aligned with buttons)

        self.title_label = QLabel("Analysis Outcome", self)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.top_container.addWidget(self.title_label)

        self.tags_container = QHBoxLayout()
        self.tags_container.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top_container.addLayout(self.tags_container)

        # Spacer between title and buttons
        self.top_container.addItem(
            QSpacerItem(
                10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        # Buttons (Square Icons with Short Hints & Mouse Hover Effect)
        self.variation_button = self.create_icon_button(
            "+", "#007bff", "New variation", self.create_variation
        )
        self.edit_button = self.create_icon_button(
            "✎", "#d39e00", "Edit", self.edit_run
        )
        self.delete_button = self.create_icon_button(
            "🗑", "#dc3545", "Delete", self.delete_run
        )

        self.open_button = self.create_icon_button(
            "To browser",
            "#7c2d12",
            "Opens the analysi result in your browser",
            self.open_in_browser,
        )
        self.open_button.setFixedWidth(100)
        self.close_button = self.create_icon_button(
            "Close", "#7c2d12", "Back to main menu", self.switch_to_dashboard
        )
        self.close_button.setFixedWidth(100)

        self.top_container.addWidget(self.variation_button)
        self.top_container.addWidget(self.edit_button)
        self.top_container.addWidget(self.delete_button)
        self.top_container.addWidget(self.open_button)
        self.top_container.addWidget(self.close_button)

        self.layout.addLayout(self.top_container)

        # --- INFO BOX (new) ---
        info_container = QVBoxLayout()
        info_container.setAlignment(Qt.AlignmentFlag.AlignTop)

        def make_info_box(html_content):
            frame = QWidget(self)
            frame.setObjectName("InfoBox")
            frame.setStyleSheet(
                """
                        QWidget#InfoBox {
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
            layout.setContentsMargins(14, 14, 18, 14)
            layout.addWidget(label)
            return frame

        fairness_html = """
                <p><b>Fairness does not end after producing AI outputs.</b></p>
                 💡 Continue interacting with stakeholders to assert that their idea of fairness is correctly implemented.
                 <br>💡 Monitor the outputs of deployed systems by rerunning the analysis on updated models and datasets.
                 <br>💡 Test model and dataset variations for multiple sensitive characteristics and parameters.
                <div style='margin-top: 10px;'>
                <p>
                Keep a balance between justifying outputs as part of a fair process and accommodating constructive criticism.
                Do not over-rely on technical justification.
                </p>
                </div>
                """

        info_container.addWidget(make_info_box(fairness_html))
        self.layout.addLayout(info_container)
        # --- END INFO BOX ---

        # Tags container (Left-aligned)

        # Results Viewer
        self.results_viewer = QWebEngineView(self)
        size_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.results_viewer.setSizePolicy(size_policy)
        self.layout.insertWidget(self.layout.count() - 2, self.results_viewer)

        self.setLayout(self.layout)

    def open_in_browser(self):
        run = self.runs[-1]
        results = run.get("analysis", dict()).get("return", "No results available.")
        with open("temp.html", "w", encoding="utf-8") as file:
            file.write(results)
        try:
            import webbrowser

            webbrowser.open_new("temp.html")
        except:
            pass

    def switch_to_dashboard(self):
        self.stacked_widget.slideToWidget(0)

    def showEvent(self, event):
        super().showEvent(event)

        # Update title and results
        if self.runs:
            run = self.runs[-1]
            self.title_label.setText(format_run(run))
            html_content = run.get("analysis", dict()).get(
                "return", "<p>No results available.</p>"
            )
            self.update_tags(run)  # Update tags
        else:
            html_content = "<p>No results available.</p>"

        # Use QTimer to ensure WebEngineView renders properly
        self.results_viewer.setHtml(
            """
        <div style="
            height:100vh;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
          <h3>
            Results too complicated to render here.<br>
            Move them <i>to browser</i> instead.
          </h3>
        </div>
        """
        )
        QTimer.singleShot(1, lambda: self.results_viewer.setHtml(html_content))
        self.results_viewer.show()

    def update_tags(self, run):
        """Refresh tags displayed below the title."""
        # Clear existing tags
        while self.tags_container.count():
            item = self.tags_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get tags
        tags = []
        if "dataset" in run:
            tags.append(run["dataset"]["module"])
        if "model" in run:
            tags.append(run["model"]["module"])
        if "analysis" in run:
            tags.append(run["analysis"]["module"])

        for tag in tags:
            self.tags_container.addWidget(
                self.create_tag_button(
                    f" {tag} ",
                    "Module info",
                    lambda checked, t=tag: self.show_tag_description(t),
                )
            )

    def show_tag_description(self, tag):
        dialog = QDialog()
        dialog.setWindowTitle("Module info")
        layout = QVBoxLayout(dialog)
        browser = QWebEngineView(self)
        browser.setFixedHeight(300)
        browser.setFixedWidth(800)
        # Example inline CSS and image
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

    def edit_run(self):
        if not self.runs:
            return
        reply = QMessageBox.question(
            self,
            "Edit?",
            f"Change modules and modify parameters of the analysis. "
            "However, this will also remove the results presented here. Consider creating a variation if you want to preserve current results.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.stacked_widget.slideToWidget(1)

    def create_variation(self):
        if not self.runs:
            return
        new_run = self.runs[-1].copy()
        new_run["status"] = "new"
        new_run["timestamp"] = now()
        self.runs[-1] = new_run
        self.dataset.append(new_run)
        self.stacked_widget.slideToWidget(1)

    def delete_run(self):
        if not self.runs:
            return
        reply = QMessageBox.question(
            self,
            "Delete?",
            f"Will permanently remove this analysis and its outcome.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            last_run = self.runs[-1]
            if last_run in self.dataset:
                self.dataset.remove(last_run)
            self.stacked_widget.slideToWidget(0)
            save_all_runs("history.json", self.dataset)
