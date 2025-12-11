from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from datetime import datetime
from mammoth_commons.externals import prepare_html
from .step import save_all_runs
from .style import Styled
from .cache import ExternalLinkPage


def format_run(run):
    return run["description"] + " " + run["timestamp"]


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


class Results(Styled):
    def create_top_container(self):
        action_bar = QHBoxLayout()
        action_bar.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.title_label = QLabel("Analysis outcome", self)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        action_bar.addWidget(self.title_label)
        self.tags_container = QHBoxLayout()
        self.tags_container.setAlignment(Qt.AlignmentFlag.AlignLeft)
        action_bar.addLayout(self.tags_container)
        action_bar.addItem(
            QSpacerItem(
                10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        action_bar.addWidget(
            self.new_action("+", "#007bff", "New variation", self.create_variation)
        )
        # action_bar.addWidget(self.new_action("✎", "#d39e00", "Edit", self.edit_run))
        # action_bar.addWidget(self.new_action("🗑", "#dc3545", "Delete", self.delete_run))
        action_bar.addWidget(
            self.new_action(
                "📃",
                "#222222",
                "Open results in your browser",
                self.open_in_browser,
            )
        )
        action_bar.addWidget(
            self.new_action(
                "X",
                "#222222",
                "Back to dashboard",
                self.switch_to_dashboard,
            )
        )
        return action_bar

    def __init__(self, stacked_widget, runs, tag_descriptions, dataset):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs
        self.dataset = dataset
        self.tag_descriptions = tag_descriptions

        info_container = QVBoxLayout()
        info_container.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(self.create_top_container())
        self.layout.addLayout(info_container)
        self.results_viewer = QWebEngineView(self)
        self.results_viewer.setZoomFactor(0.8)
        self.layout.insertWidget(self.layout.count() - 1, self.results_viewer)
        self.results_viewer.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
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

    def switch_to_restart(self):
        self.stacked_widget.slideToWidget(1)

    def showEvent(self, event):
        super().showEvent(event)
        self.results_viewer.setHtml(
            """
            <div style="height:100vh; display:flex; align-items:center; justify-content:center; text-align:center;">
              <h3> Results too complicated to render here.<br>Move them <i>to browser</i> instead.</h3>
            </div>
            """
        )
        if self.runs:
            run = self.runs[-1]
            self.title_label.setText(format_run(run))
            html_content = run.get("analysis", dict()).get(
                "return", "<p>No results available.</p>"
            )
            self.update_tags(run)
        else:
            html_content = "<p>No results available.</p>"
        QTimer.singleShot(
            1,
            lambda: self.results_viewer.setHtml(
                prepare_html(html_content), QUrl("file:///")
            ),
        )
        self.results_viewer.show()

    def update_tags(self, run):
        return
        while self.tags_container.count():
            item = self.tags_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        tags = []
        if "dataset" in run:
            tags.append(run["dataset"]["module"])
        if "model" in run:
            tags.append(run["model"]["module"])
        if "analysis" in run:
            tags.append(run["analysis"]["module"])
        for tag in tags:
            self.tags_container.addWidget(
                self.new_tag(
                    f" {tag} ",
                    "Module info",
                    lambda checked, t=tag: self.show_tag_description(t),
                )
            )

    def show_tag_description(self, tag):
        dialog = QDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog.setWindowTitle("Module info")
        layout = QVBoxLayout(dialog)
        browser = QWebEngineView(self)
        browser.setFixedHeight(800)
        browser.setFixedWidth(800)
        # Example inline CSS and image
        html = self.tag_descriptions.get(tag, "No description available.")
        html = f"""
        <html>
        <head>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{font-family: Arial, sans-serif;font-size: 14px;color: #333;background-color: #white;padding: 10px;}}
            h1 {{font-size: 18px;color: #0055aa;}}
            img {{max-width: 100%;border: 1px solid #ccc;border-radius: 4px;}}
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

    def edit_run(self):
        if not self.runs:
            return
        if (
            QMessageBox.question(
                self,
                "Edit?",
                f"Change modules and modify parameters of the analysis. "
                "However, this will also remove the results presented here. Consider creating a variation if you want to preserve current results.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
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
        if (
            QMessageBox.question(
                self,
                "Delete?",
                f"Will permanently remove this analysis and its outcome.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        last_run = self.runs[-1]
        if last_run in self.dataset:
            self.dataset.remove(last_run)
        self.stacked_widget.slideToWidget(0)
        save_all_runs("history.json", self.dataset)
