from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QMutex
from mai_bias.backend.loaders import registry
from mai_bias.states.step import Step, save_all_runs, InfoBox
from mammoth_commons import integration_callback

global items


class ModelLoaderThread(QThread):
    finished_success = Signal(object)
    finished_failure = Signal(str)
    canceled = Signal()
    notify = Signal(str)

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._is_canceled = False
        self.mutex = QMutex()
        self.setTerminationEnabled(True)
        integration_callback.register_progress_callback(self.progress_callback)

    def progress_callback(self, progress, message):
        progress = min(1, max(progress, 0))
        progress_blocks = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        full_blocks = int(progress * 40)
        partial_block_index = int((progress * 40 - full_blocks) * len(progress_blocks))
        bar = "█" * full_blocks
        if full_blocks < 40:
            bar += progress_blocks[partial_block_index]
        bar += "&nbsp;" * (40 - len(bar))
        self.mutex.lock()
        self.notify.emit(
            "<span style='color: blue; background: lightgray; font-family: monospace;'>"
            + bar
            + "</span><br>"
            + message
        )
        self.mutex.unlock()

    def run(self):
        try:
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            self.pipeline["model"]["return"] = registry.name_to_runnable[
                self.pipeline["model"]["module"]
            ](**self.pipeline["model"]["params"])
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            self.finished_success.emit(self.pipeline)
        except Exception as e:
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()
        self.terminate()


class SelectModel(Step):
    def __init__(self, step_name, stacked_widget, dataset_loaders, runs, dataset):
        super().__init__(step_name, stacked_widget, dataset_loaders, runs, dataset)
        info_box = InfoBox(
            """
        <p>
        Fairness is a consideration at each step during the lifecycle of an AI system; 
        it spans fair design, development interventions, and ongoing practices to maintain quality. 
        Starting from the design phase, determine a desired outcome and build or investigate your system with that in mind:
        </p>
    
        💡 <b>Weak fairness</b> passively debiases predictions.<br/>
        💡 <b>Strong fairness</b> actively participates in societal improvement 
        (more access, opportunities, life chances to all people, etc.).
    
        <p>
        Consider an AI system that regulates university admissions <b>[1]</b>. 
        Weak fairness aims to correct biases related to several intersecting protected attributes,
        such as ethnicity, gender, disability, or national origin.
        Forms of strong fairness could include correcting the underadmission of
        certain groups in previous years, or placing equal importance on both more and less
        affordable extracurricular activities that influence access to universities, 
        given that some groups struggle to pay for expensive ones <b>[2]</b>.
        </p>
    
        <p style="font-style: italic; color: #555;">
        [1] Costanza-Chock, Sasha. “Design Justice. Community-led practices to
        build the worlds we need”, Cambridge, MA: The MIT Press (2020)
        <br/>
        [2] Giovanola, Benedetta, and Simona Tiribelli. 
        "Weapons of moral construction? On the value of fairness in algorithmic decision-making."
        <i>Ethics and Information Technology</i> 24, no. 1: 3 (2022)
        </p>
        """,
            self,
        )
        self.layout().insertWidget(self.layout().count() - 2, info_box)

    def showEvent(self, event):
        pipeline = self.runs[-1]
        self.description_input.setText(pipeline["description"])
        module = pipeline["dataset"]["module"]
        loaders = [
            loader
            for loader, values in registry.model_loaders.items()
            if module in values["compatible"]
        ]
        self.dataset_selector.clear()
        self.dataset_selector.addItems(["Select a model loader"] + loaders)
        self.defaults = self.runs[-1].get("model", dict()).get("params", dict())
        # self.update_param_form(self.runs[-1].get("model", dict()).get("module", "Select a model loader"))
        self.dataset_selector.setCurrentIndex(
            self.dataset_selector.findText(
                self.runs[-1]
                .get("model", dict())
                .get("module", "Select a model loader")
            )
        )
        super().showEvent(event)

    def next(self):
        self.save("model")
        save_all_runs("history.json", self.dataset)
        pipeline = self.runs[-1]

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Loading model")
        self.loading_message.setText("Please wait while the model is loading...")
        self.loading_message.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.StandardButton.Cancel).clicked.connect(
            self.cancel_loading
        )
        self.loading_message.show()

        # Start the model loading thread using QThread
        self.thread = ModelLoaderThread(pipeline)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.notify.connect(self.on_notify)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.slideToWidget(3)

    def on_notify(self, message):
        self.loading_message.setText(message)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("model")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.slideToWidget(0)
        save_all_runs("history.json", self.dataset)

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        event.accept()
