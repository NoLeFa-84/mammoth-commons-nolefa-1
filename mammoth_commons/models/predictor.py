from mammoth_commons.models.model import Model


class Predictor(Model):
    def __init__(self):
        super().__init__()

    def predict(self, dataset, sensitive):
        raise Exception(f"{self.__class__.__name__} has no method predict method")
