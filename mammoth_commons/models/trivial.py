from mammoth_commons.datasets import Labels
from mammoth_commons.models.predictor import Predictor
from numpy import ones, zeros, sum


class TrivialPredictor(Predictor):
    def __init__(self):
        super().__init__()

    def predict(self, dataset, sensitive: list[str]):
        dataset = dataset.to_csv(sensitive)
        labels = dataset.labels
        counts = {label: sum(labels[label]) for label in labels}
        const = max(counts, key=counts.get)
        n = len(labels[const])
        return Labels({l: ones((n,)) if l == const else zeros((n,)) for l in labels})
