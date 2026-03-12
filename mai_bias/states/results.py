import os

from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QMessageBox,
    QDialog,
    QFileDialog,
)
from PySide6.QtCore import Qt, QTimer, QUrl, QByteArray, QSize
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


def icon_from_svg(svg: str, size: int = 20) -> QIcon:
    ba = QByteArray(svg.encode("utf-8"))
    pix = QPixmap()
    if not pix.loadFromData(ba, "SVG"):
        raise RuntimeError("Could not parse SVG")
    pix = pix.scaled(QSize(size, size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pix)


SVG_CLONE = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#222222" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14l4-4h9c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/></svg>"""
SVG_GLOBE = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#222222" d="M12 2a10 10 0 0 0-8 4.3c2 .8 5 .7 6.5-.4 1.6-1.2 1.9-2.9 1.5-3.9zm0 20a10 10 0 0 0 8-4.3c-2-.8-5-.7-6.5.4-1.6 1.2-1.9 2.9-1.5 3.9zM2 12a10 10 0 0 0 4.3 8c.8-2 .7-5-.4-6.5-1.2-1.6-2.9-1.9-3.9-1.5z"/></svg>"""
SVG_SAVE = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path fill="#222222"
        d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81c-.54-.5-1.25-.81-2.04-.81-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.15c-.05.21-.09.43-.09.66 0 1.66 1.34 3 3 3s3-1.34 3-3-1.34-3-3-3z"/>
</svg>"""
SVG_CLOSE = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M18 6 L6 18 M6 6 L18 18"
          fill="none"
          stroke="#222222"
          stroke-width="2"
          stroke-linecap="round"/>
</svg>
"""


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
        action_bar.addWidget(
            self.new_action(
                " New variation",
                "#EEEEEE",
                "Create a copy where you can change parameters",
                self.create_variation,
                width=150,
                icon=icon_from_svg(SVG_CLONE),
            )
        )
        # action_bar.addWidget(self.new_action("✎", "#d39e00", "Edit", self.edit_run))
        # action_bar.addWidget(self.new_action("🗑", "#dc3545", "Delete", self.delete_run))
        action_bar.addWidget(
            self.new_action(
                " Open in browser",
                "#EEEEEE",
                "Open results in your browser",
                self.open_in_browser,
                width=150,
                icon=icon_from_svg(SVG_GLOBE),
            )
        )
        action_bar.addWidget(
            self.new_action(
                " Save and share",
                "#EEEEEE",
                "Save as an html file that can be shared",
                self.save_as,
                width=150,
                icon=icon_from_svg(SVG_SAVE),
            )
        )
        action_bar.addItem(
            QSpacerItem(
                10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        action_bar.addWidget(
            self.new_action(
                "Back",
                "#EEEEEE",
                "Back to dashboard",
                self.switch_to_dashboard,
                width=100,
                icon=icon_from_svg(SVG_CLOSE),
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

    def save_as(self) -> None:
        """
        Let the user pick a location and save the HTML results of the latest
        run to a file.

        The dialog defaults to the user's “Documents” folder and suggests a
        filename based on the run’s timestamp/description.  The saved file
        contains the fully‑prepared HTML (including the stylesheet injected by
        ``prepare_html``).
        """
        if not self.runs:
            QMessageBox.information(
                self,
                "No results",
                "There is no analysis result to save.",
            )
            return
        run = self.runs[-1]
        html_content = run.get("analysis", {}).get(
            "return", "<p>No results available.</p>"
        )
        # html_to_save = prepare_html(html_content) # DO NOT DO THIS AS WE NEED TO AVOID THE CACHE FOR TRANSPORTABILITY
        html_to_save = html_content
        timestamp = run.get("timestamp", now()).replace(":", "-").replace(" ", "_")
        default_name = f"run-{timestamp}.html"
        # cannot be determined).
        start_dir = os.path.expanduser("~/Documents")
        if not os.path.isdir(start_dir):
            start_dir = os.getcwd()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save analysis as a file that you can share.",
            filter="HTML Files (*.html);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.endswith(".html"):
            file_path += ".html"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_to_save)
            QMessageBox.information(
                self,
                "Saved",
                f"The analysis result was saved to the following file."
                f"You can share this with other people, and they can open it in their browser:\n{file_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Save failed",
                f"Could not write the file:\n{file_path}\n\nError: {exc}",
            )

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
        self.results_viewer.setHtml("""
            <div style="height:100vh; display:flex; align-items:center; justify-content:center; text-align:center;">
              <h3> Results too complicated to render here.<br>Move them <i>to browser</i> instead.</h3>
            </div>
            """)
        if self.runs:
            run = self.runs[-1]
            self.title_label.setText("")  # format_run(run))
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
