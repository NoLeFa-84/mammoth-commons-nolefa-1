from mammoth_commons.models.predictor import Predictor
import re


class ONNXEnsemble(Predictor):
    def __init__(
        self,
        models,
        _=None,
        alphas=None,
        classes=None,
        n_classes=None,
        theta=None,
        pareto=None,
        sensitives=None,
        **kwargs,
    ):
        super().__init__()
        from mmm_fair_cli.onnx_utils import ONNX_MMM

        assert (
            _ is None
        ), "Internal error: ONNXEnsemble was accidentally constructed with more positional arguments than acceptable"
        self.mmm = ONNX_MMM(
            models, alphas, classes, n_classes, theta, pareto, sensitives
        )

    def _extract_number(self, filename):
        match = re.search(r"_(\d+)\.onnx$", filename)
        return int(match.group(1)) if match else float("inf")

    def predict(self, dataset, sensitive, theta=None):
        """assert (
            sensitive is None or len(sensitive) == 0
        ), "ONNXEnsemble can only be called with no declared sensitive attributes" """

        preds = self.mmm.predict(dataset, sensitive, theta)

        return preds
