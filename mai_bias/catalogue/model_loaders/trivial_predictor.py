from mammoth_commons.datasets import Labels
from mammoth_commons.models import Predictor
from mammoth_commons.integration import loader
from numpy import ones, zeros, sum


class TrivialPredictor(Predictor):
    def predict(self, dataset, sensitive: list[str]):
        dataset = dataset.to_csv(sensitive)
        labels = dataset.labels
        counts = {label: sum(labels[label]) for label in labels}
        const = max(counts, key=counts.get)
        n = len(labels[const])
        return Labels({l: ones((n,)) if l == const else zeros((n,)) for l in labels})


@loader(namespace="mammotheu", version="v053", python="3.13", packages=())
def model_trivial_predictor() -> TrivialPredictor:
    """Creates a trivial predictor that returns the most common predictive label value among provided data.
    If the label is numeric, the median is computed instead. This model servers as an informed baseline
    of what happens even for an uninformed predictor. Several kinds of class biases may exist, for example
    due to different class imbalances for each sensitive attribute dimension (e.g., for old white men
    compared to young hispanic women)."""
    return TrivialPredictor()
