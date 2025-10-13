from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QMutex
from mai_bias.backend.loaders import registry
from mai_bias.states.step import Step, save_all_runs, InfoBox
import traceback
from mammoth_commons import integration_callback


class AnalysisThread(QThread):
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
            args = self.pipeline["analysis"]["params"]
            dataset = self.pipeline["dataset"]["return"]
            model = self.pipeline["model"]["return"]
            sensitive = args.get("sensitive", "")
            if "," in sensitive:
                sensitive = sensitive.split(",")
            elif sensitive == "":
                sensitive = []
            else:
                sensitive = [sensitive]
            sensitive = [s.strip() for s in sensitive]
            args = {
                k: v
                for k, v in args.items()
                if k not in ["dataset", "model", "sensitive"]
            }
            self.pipeline["analysis"]["return"] = registry.name_to_runnable[
                self.pipeline["analysis"]["module"]
            ](dataset, model, sensitive, **args).text()
            self.pipeline["dataset"]["return"] = None
            self.pipeline["model"]["return"] = None
            self.pipeline["status"] = "completed"

            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()

            self.finished_success.emit(self.pipeline)
        except Exception as e:
            traceback.print_exception(e)
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()
        self.terminate()


class SelectAnalysis(Step):
    def __init__(self, step_name, stacked_widget, dataset_loaders, runs, dataset):
        super().__init__(step_name, stacked_widget, dataset_loaders, runs, dataset)
        info_box = InfoBox(
            """
<p>💡 <b>Fairness is context-specific.</b> There is no general fairness definition that applies to every context or use case.
This page lets you select fairness/bias assessment methodologies that contain definitions from the computer science literature. 
However, which ones are suitable depends on the specific situation you are studying; less common methodologies and definitions 
could be preferable in certain cases.</p>

<p>💡 <b>There can be conflicting interests and opinions on what is fair.</b> When different stakeholders with different ideas on what 
constitutes a fair solution to a problem are involved, fairness becomes the result of a negotiation process that is affected 
by power relations. Think of an example AI system that evaluates loan requests: bank clients might want their personal circumstances to
be part of the evaluation, but lenders might think it is fair to provide impartial and systematic responses (although these may also contain
biases that were not accounted for during system creation, like historical racism in training data).
</p>
        """,
            self,
        )
        self.layout().insertWidget(self.layout().count() - 2, info_box)

    def showEvent(self, event):
        pipeline = self.runs[-1]
        self.description_input.setText(pipeline["description"])
        compatible_methods = [
            method
            for method, entries in registry.analysis_methods.items()
            if issubclass(
                registry.parameters_to_class[pipeline["dataset"]["module"]]["return"],
                registry.parameters_to_class[method][entries["parameters"][0][0]],
            )
            and issubclass(
                registry.parameters_to_class[pipeline["model"]["module"]]["return"],
                registry.parameters_to_class[method][entries["parameters"][1][0]],
            )
        ]
        self.dataset_selector.clear()
        self.dataset_selector.addItems(
            ["Select a fairness analysis method"] + compatible_methods
        )
        self.defaults = self.runs[-1].get("analysis", dict()).get("params", dict())
        # self.update_param_form(self.runs[-1].get("analysis", dict()).get("module", "Select a fairness analysis method"))
        self.dataset_selector.setCurrentIndex(
            self.dataset_selector.findText(
                self.runs[-1]
                .get("analysis", dict())
                .get("module", "Select a fairness analysis method")
            )
        )
        super().showEvent(event)

    def next(self):
        self.save("analysis")
        pipeline = self.runs[-1]

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Running fairness analysis")
        self.loading_message.setText(
            "Please wait while the fairness analysis is running..."
        )
        self.loading_message.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.StandardButton.Cancel).clicked.connect(
            self.cancel_loading
        )
        self.loading_message.show()

        # Start the analysis thread
        self.thread = AnalysisThread(pipeline)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.notify.connect(self.on_notify)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.slideToWidget(4)
        save_all_runs("history.json", self.dataset)

    def on_notify(self, message):
        self.loading_message.setText(message)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)
        save_all_runs("history.json", self.dataset)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("analysis")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.slideToWidget(0)
        save_all_runs("history.json", self.dataset)

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        event.accept()
