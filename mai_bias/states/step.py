from PySide6.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QFileDialog,
    QDialog,
    QListWidget,
    QScrollArea,
)
from PySide6.QtGui import QIntValidator, QDoubleValidator
from mammoth_commons.externals import prepare_html
from PySide6.QtCore import Qt, QLocale, QSize
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QWidget,
    QToolButton,
)
from PySide6.QtGui import QIcon, QPixmap
from mammoth_commons.externals import prepare, pd_read_csv
import json
import os
import csv
import mammoth_commons.externals
from .style import Styled
from .step_utils.card_button import CardButton


def save_all_runs(path, runs):
    copy_runs = list()
    for run in runs:
        copy_run = dict()
        copy_run["timestamp"] = run["timestamp"]
        copy_run["description"] = run["description"]
        copy_run["status"] = run.get("status", None)
        if "dataset" in run:
            copy_run["dataset"] = {
                "module": run["dataset"]["module"],
                "params": run["dataset"]["params"],
            }
        if "model" in run:
            copy_run["model"] = {
                "module": run["model"]["module"],
                "params": run["model"]["params"],
            }
        if "analysis" in run:
            copy_run["analysis"] = {
                "module": run["analysis"]["module"],
                "params": run["analysis"]["params"],
                "return": run["analysis"].get("return", None),
            }
        copy_runs.append(copy_run)
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(copy_runs))


def load_all_runs(path):
    if not os.path.exists(path):
        return list()
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_name(name):
    """Format parameter names for better display."""
    return name.replace("_", " ").capitalize()


class InfoBox(QFrame):
    def __init__(self, html_content, parent=None):
        super().__init__(parent)
        self.setObjectName("InfoBox")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame#InfoBox {background-color: #dddddd; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px;}
            QLabel {color: #334155; font-size: 13px; line-height: 1.4em;}
            a {color: #0369a1; text-decoration: none; font-weight: 600;}
            a:hover {text-decoration: underline;}
            ul {margin-left: 16px;}
            li {margin: 4px 0;}
            """
        )

        label = QLabel(html_content, self)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)


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
        scroll.setStyleSheet("background:transparent; border: none;")

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


class Step(Styled):
    def __init__(self, step_name, stacked_widget, dataset_loaders, runs, dataset):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.dataset_loaders = dataset_loaders
        self.first_selection = True
        self.runs = runs
        self.dataset = dataset
        self.show_all_params = False

        layout = QVBoxLayout()

        self.label = QLabel(step_name, self)
        self.label.setStyleSheet("font-size:32px;font-weight:bold")
        layout.addWidget(self.label, 0)

        selector_row = QWidget()
        selector_layout = QHBoxLayout(selector_row)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(6)

        self.dataset_selector = ScrollSelector(
            ["Select a module"] + list(dataset_loaders.keys()),
            specs=dataset_loaders,
            on_change=self.update_param_form,
            parent=self,
        )
        self.dataset_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        icon_path = prepare(
            "https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/params.png?raw=true"
        )
        icon = QIcon(QPixmap(icon_path))

        self.param_toggle_button = QToolButton(self)
        self.param_toggle_button.setCheckable(True)
        self.param_toggle_button.setIcon(icon)
        self.param_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        self.param_toggle_button.setIconSize(QSize(72, 72))
        self.param_toggle_button.setFixedSize(96, 96)
        self.param_toggle_button.setStyleSheet(
            "QToolButton{background:#eee;border-radius:6px;padding:4px}QToolButton:hover{background:#d0d0d0;border: 1px solid #cccccc}"
        )
        self.param_toggle_button.clicked.connect(self.toggle_param_visibility)
        self.param_toggle_button.hide()

        icon_path = prepare(
            "https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/warning.png?raw=true"
        )
        icon = QIcon(QPixmap(icon_path))
        self.warnings_toggle_button = QToolButton(self)
        self.warnings_toggle_button.setCheckable(True)
        self.warnings_toggle_button.setIcon(icon)
        self.warnings_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        self.warnings_toggle_button.setIconSize(QSize(72, 72))
        self.warnings_toggle_button.setFixedSize(96, 96)
        self.warnings_toggle_button.setText("responsibility")
        self.warnings_toggle_button.setStyleSheet(
            "QToolButton{background:#eee;border-radius:6px;padding:4px}QToolButton:hover{background:#d0d0d0;border: 1px solid #cccccc}"
        )
        self.warnings_toggle_button.clicked.connect(self.toggle_param_visibility)
        self.warnings_toggle_button.hide()

        selector_layout.addWidget(self.dataset_selector, 1)
        selector_layout.addWidget(
            self.param_toggle_button, 0, Qt.AlignmentFlag.AlignTop
        )
        selector_row.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(selector_row, 1)
        # selector_row.setMinimumHeight(400)
        # layout.addStretch(2)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color:#444;margin:6px 0;")
        layout.addWidget(separator, 0)

        content_layout = QVBoxLayout()
        self.param_form = QFormLayout()
        self.param_inputs = {}

        self.form_widget = QWidget()
        self.form_widget.setLayout(self.param_form)
        self.form_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        left_col = QVBoxLayout()
        label = QLabel("Configure", self)
        label.setStyleSheet("font-size:32px;font-weight:bold")
        left_col.addWidget(label)
        left_col.addWidget(self.form_widget)
        right_col = QHBoxLayout()
        right_col.addWidget(self.param_toggle_button, 0, Qt.AlignmentFlag.AlignBottom)
        right_col.addWidget(
            self.warnings_toggle_button, 0, Qt.AlignmentFlag.AlignBottom
        )
        container_row = QHBoxLayout()
        container_row.addLayout(left_col, 1)
        container_row.addLayout(right_col, 0)
        content_layout.addLayout(container_row)

        layout.addLayout(content_layout, 0)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color:#444;margin:6px 0;")
        content_layout.addWidget(separator, 0)
        content_layout.addSpacing(32)

        button_layout = QHBoxLayout()
        next_button = QPushButton(
            "Run" if hasattr(self, "switch_to_restart") else "Next", self
        )
        next_button.setStyleSheet(
            "QPushButton{background:#07f;color:#fff;border-radius:5px;padding:6px}QPushButton:hover{background:#059}"
        )
        next_button.clicked.connect(self.next)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.setStyleSheet(
            "QPushButton{background:#d33;color:#fff;border-radius:5px}QPushButton:hover{background:#a00}"
        )
        cancel_button.setFixedSize(80, 30)
        cancel_button.clicked.connect(self.switch_to_dashboard)

        button_layout.addWidget(next_button)
        if hasattr(self, "switch_to_restart"):
            self.restart_button = QPushButton("Edit pipeline", self)
            self.restart_button.setStyleSheet(
                "QPushButton{background:#d33;color:#fff;border-radius:5px}QPushButton:hover{background:#a00}"
            )
            self.restart_button.setFixedSize(80, 30)
            self.restart_button.clicked.connect(self.switch_to_restart)
            button_layout.addWidget(self.restart_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.defaults = {}
        self.update_param_form(self.dataset_selector.currentText())

    def update_param_form(self, dataset_name):
        if self.first_selection and dataset_name != self.dataset_selector.itemText(0):
            if dataset_name not in self.dataset_loaders:
                self.dataset_selector.removeItem(0)
            self.first_selection = False
        for i in reversed(range(self.param_form.rowCount())):
            self.param_form.removeRow(i)
        self.param_inputs.clear()
        if dataset_name not in self.dataset_loaders:
            return
        loader = self.dataset_loaders[dataset_name]
        self.last_url = None
        self.last_delimiter = None
        self.count_hidden_params = 0
        for name, param_type, default, description in loader["parameters"]:
            can_be_hidden = name != "sensitive" and default != "" and default != "None"
            if can_be_hidden:
                self.count_hidden_params += 1
            default = self.defaults.get(name, default)
            if name == "dataset" or name == "model":
                continue
            param_options = loader.get("parameter_options", {}).get(name, [])
            param_widget = self.create_input_widget(
                name, param_type, default, description, param_options
            )
            if can_be_hidden and not self.show_all_params:
                param_widget.hide()
            self.param_form.addRow(param_widget)
        if self.count_hidden_params:
            self.param_toggle_button.show()
        else:
            self.param_toggle_button.hide()
        self.param_toggle_button.setText(
            "hide details"
            if self.show_all_params
            else "for experts"  # f"{self.count_hidden_params}"
        )

    def toggle_param_visibility(self):
        self.show_all_params = self.param_toggle_button.isChecked()
        self.update_param_form(self.dataset_selector.currentText())

    def open_sensitive_modal(self, title, input_field, columns):
        if not isinstance(columns, list):
            path = columns[0].text()
            delimiter = columns[1].text() if columns[1] is not None else None
            if len(path) == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    "The previous file was empty and could not be used as reference.",
                )
                return
            if delimiter is not None and len(delimiter) == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    "The previous file's delimiter was empty and could not be used as reference.</b>",
                )
                return
            try:
                if delimiter is None:
                    try:
                        with open(path, "r") as file:
                            sample = file.read(4096)
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample).delimiter
                            delimiter = str(delimiter)
                            import string

                            if delimiter in string.ascii_letters:
                                common_delims = [",", ";", "|", "\t"]
                                counts = {d: sample.count(d) for d in common_delims}
                                delimiter = (
                                    max(counts, key=counts.get)
                                    if any(counts.values())
                                    else ","
                                )
                    except Exception:
                        delimiter = ","
                df = pd_read_csv(
                    path, nrows=3, on_bad_lines="skip", delimiter=delimiter
                )
                columns = df.columns.tolist()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not read the previous file to use as reference:<br><b>{str(e)}</b>",
                )
                return

        prev_value = input_field.text()
        prev_selection = set(prev_value.split(","))

        dialog = QDialog(self)
        dialog.setStyleSheet("background-color: white;")
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        layout = QVBoxLayout()
        list_widget = QListWidget(dialog)
        list_widget.addItems(columns)
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(list_widget)

        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.text() in prev_selection:
                item.setSelected(True)

        def cancel():
            input_field.setText(prev_value)
            dialog.accept()

        cancel_button = QPushButton("Cancel", dialog)
        cancel_button.clicked.connect(cancel)
        layout.addWidget(cancel_button)

        confirm_button = QPushButton("Done", dialog)
        confirm_button.clicked.connect(
            lambda: self.set_sensitive_values(dialog, list_widget, input_field)
        )
        layout.addWidget(confirm_button)

        dialog.setLayout(layout)
        dialog.exec()

    def set_sensitive_values(self, dialog, list_widget, input_field):
        selected_items = [item.text() for item in list_widget.selectedItems()]
        input_field.setText(", ".join(selected_items))
        dialog.accept()

    def create_input_widget(
        self, name, param_type, default, description, param_options
    ):
        if name == "sensitive":
            description = "<h1>Sensitive/protected attributes</h1>Protected attributes usually refer to personal characteristics protected by the law on non-discrimination. For example, the EU Charter of Fundamental Rights in Article 21 enacts a non-exhaustive list of grounds for non-discrimination as follows: sex, race, colour, ethnic or social origin, genetic features, language, religion or belief, political or any other opinion, membership of a national minority, property, birth, disability, age or sexual orientation.<br><br>Here we include also attributes that are not necessarily protected by the law, but that can still lead to potential forms of discrimination. "
        param_layout = QHBoxLayout()
        param_layout.setContentsMargins(0, 0, 0, 0)

        helper = None
        preview = None
        if "layer" in name:
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            if self.runs[-1].get("model", dict()).get("return", None) is not None:
                select_button = QPushButton("...")
                select_button.setToolTip("Select from options")
                select_button.setFixedSize(30, 20)
                select_button.setStyleSheet(
                    f"""QPushButton {{
                        background-color: #dddd88; 
                        border-radius: 5px;
                    }}
                    QPushButton:hover {{
                        background-color: {self.highlight_color('#dddd88')};
                    }}"""
                )
                select_button.clicked.connect(
                    lambda: self.open_sensitive_modal(
                        f"Select {name}",
                        input_widget,
                        mammoth_commons.externals.get_model_layer_list(
                            self.runs[-1].get("model", dict()).get("return", None)
                        ),
                    )
                )
                helper = select_button
        elif (
            "numeric" in name
            or "categorical" in name
            or "label" in name
            or "target" in name
            or "ignored" in name
            or "attribute" in name
        ):
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            if self.last_url is not None:
                select_button = QPushButton("...")
                select_button.setToolTip("Select from options")
                select_button.setFixedSize(30, 20)
                select_button.setStyleSheet(
                    f"""QPushButton {{background-color: #dddd88; border-radius: 5px;}}
                    QPushButton:hover {{background-color: {self.highlight_color('#dddd88')};}}"""
                )
                last_url = self.last_url
                last_delimiter = self.last_delimiter
                select_button.clicked.connect(
                    lambda: self.open_sensitive_modal(
                        f"Select {name} columns",
                        input_widget,
                        (last_url, last_delimiter),
                    )
                )
                helper = select_button
        elif "library" in name or "libraries" in name:
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            if self.last_url is not None:
                select_button = QPushButton("...")
                select_button.setToolTip("Select from options")
                select_button.setFixedSize(30, 20)
                select_button.setStyleSheet(
                    f"""QPushButton {{background-color: #dddd88; border-radius: 5px;}}
                    QPushButton:hover {{background-color: {self.highlight_color('#dddd88')};}}"""
                )
                last_url = self.last_url
                select_button.clicked.connect(
                    lambda: self.open_sensitive_modal(
                        f"Select {name}",
                        input_widget,
                        mammoth_commons.externals.get_import_list(last_url.text()),
                    )
                )
                helper = select_button
        elif name == "sensitive":
            if not self.runs:
                return QWidget()
            columns = self.runs[-1]["dataset"]["return"]
            columns = (
                [""]
                if columns is None or not hasattr(columns, "cols")
                else columns.cols
            )

            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

            select_button = QPushButton("...")
            select_button.setToolTip("Select from options")
            select_button.setFixedSize(30, 20)
            select_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #dddd88; 
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.highlight_color('#dddd88')};
                }}"""
            )
            select_button.clicked.connect(
                lambda: self.open_sensitive_modal(
                    "Select sensitive attributes", input_widget, columns
                )
            )

            helper = select_button
        elif param_options:  # If parameter options are provided, use a dropdown
            input_widget = QComboBox(self)
            input_widget.addItems(param_options)
            input_widget.setCurrentText(
                default if default in param_options else param_options[0]
            )
        elif param_type == "int":
            input_widget = QLineEdit(self)
            input_widget.setValidator(QIntValidator())
            input_widget.setText(str(default) if default != "None" else "0")
        elif param_type == "float":
            input_widget = QLineEdit(self)
            validator = QDoubleValidator()
            validator.setLocale(QLocale("C"))
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            input_widget.setValidator(validator)
            input_widget.setText(str(default) if default != "None" else "0.0")
        elif param_type == "bool":
            input_widget = QCheckBox(self)
            input_widget.setChecked(str(default).lower() == "true")
        elif "dir" in name:
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

            file_button = QPushButton("...")
            file_button.setToolTip("Navigate")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #dddd88; 
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.highlight_color('#dddd88')};
                }}"""
            )
            file_button.clicked.connect(lambda: self.select_dir(input_widget))
            helper = file_button

        elif param_type == "url":
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            self.last_url = input_widget

            file_button = QPushButton("...")
            file_button.setToolTip("Navigate")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #dddd88; 
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.highlight_color('#dddd88')};
                }}"""
            )
            file_button.clicked.connect(lambda: self.select_path(input_widget))
            helper = file_button

            def preview_file():
                file_path = input_widget.text().strip()
                if not file_path:
                    QMessageBox.warning(self, "Error", "No file selected.")
                    return
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        lines = [file.readline().strip() for _ in range(20)]
                    preview_text = "\n".join(line for line in lines if line)

                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("File preview")
                    msg_box.setToolTip("Peek at the first 20 lines")
                    msg_box.setText(preview_text if preview_text else "File is empty.")
                    msg_box.setIcon(
                        QMessageBox.Icon.NoIcon
                    )  # Removes the information icon
                    msg_box.exec_()

                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Could not read the file or failed to convert it to a human-friendly format:\n{str(e)}",
                    )

            file_button = QPushButton("Preview")
            file_button.setFixedSize(50, 20)
            file_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #ddbbdd; 
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.highlight_color('#ddbbdd')};
                }}"""
            )
            file_button.clicked.connect(preview_file)
            preview = file_button

        elif "delimiter" in name:
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            last_url = self.last_url

            def recommend_delimiter():
                path = last_url.text()
                if len(path) == 0:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"The previous file was empty and could not be used as reference.",
                    )
                    return
                try:
                    with open(path, "r") as file:
                        sample = file.read(4096)
                        sniffer = csv.Sniffer()
                        delimiter = sniffer.sniff(sample).delimiter
                        delimiter = str(delimiter)
                        import string

                        if delimiter in string.ascii_letters:
                            common_delims = [",", ";", "|", "\t"]
                            counts = {d: sample.count(d) for d in common_delims}
                            # pick the one with highest count, fallback to ","
                            delimiter = (
                                max(counts, key=counts.get)
                                if any(counts.values())
                                else ","
                            )
                        input_widget.setText(delimiter)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Could not read the previous file to use as reference:<br><b>{str(e)}</b>",
                    )

            file_button = QPushButton("Find")
            file_button.setToolTip("Autodetect based on csv rules")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #dddd88; 
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.highlight_color('#dddd88')};
                }}"""
            )
            file_button.clicked.connect(recommend_delimiter)
            preview = file_button

        else:  # Default to a normal text field
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

        if input_widget is not None:
            input_widget.setStyleSheet(
                """
                QLineEdit {
                    background-color: #fff;
                    border: 0px solid #ccc;
                }
                QLineEdit:hover {
                    border: 0px solid #999;
                }
                QLineEdit:focus {
                    background-color: #fff;
                    border: 0px solid #444;
                }
                """
            )

        self.param_inputs[name] = input_widget

        label = QLabel(format_name(name))
        label.setFixedSize(150, 20)
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        help_button = QPushButton("?")
        help_button.setFixedSize(20, 20)
        help_button.setStyleSheet(
            f"""
            QPushButton {{background-color: #dddddd; border-radius: 10px;}}
            QPushButton:hover {{background-color: {self.highlight_color('#dddddd')};}}"""
        )
        help_button.setToolTip("Parameter info")
        help_button.clicked.connect(
            lambda: self.show_help_popup(format_name(name), description)
        )

        param_layout.addWidget(label)
        param_layout.addWidget(help_button)
        if helper is not None:
            param_layout.addWidget(helper)
        if preview is not None:
            param_layout.addWidget(preview)
        param_layout.addWidget(input_widget)

        param_widget = QWidget()
        param_widget.setLayout(param_layout)
        return param_widget

    def select_dir(self, input_field):
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            input_field.setText(path)

    def select_path(self, input_field):
        path = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            input_field.setText(path[0])

    def show_help_popup(self, param_name, description):
        """Show a popup window with the parameter description."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Parameter info")
        msg.setText(description)
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.exec()

    def save(self, step):
        pipeline = self.runs[-1]
        dataset_name = self.dataset_selector.currentText()
        params = {}
        for param, field in self.param_inputs.items():
            if isinstance(field, QCheckBox):
                params[param] = field.isChecked()
            elif isinstance(field, QComboBox):
                params[param] = field.currentText()
            else:
                params[param] = field.text()
        pipeline[step] = {"module": dataset_name, "params": params}
        pipeline["description"] = ""  # self.description_input.text().strip()

    def show_error_message(self, message):
        error_msg = QMessageBox(self)
        if not message:
            message = "Unknown assertion error"
        if message[0] == "'" and message[-1] == "'":
            message = message[1:-1]
        message = "The following issue must be addressed:<br><b>" + message + "</b>"
        error_msg.setWindowTitle("Error")
        error_msg.setText(message)
        error_msg.setIcon(QMessageBox.Icon.Critical)
        error_msg.setModal(True)
        error_msg.exec()
