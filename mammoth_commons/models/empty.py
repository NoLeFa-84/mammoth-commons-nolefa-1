from mammoth_commons.models.model import Model


class EmptyModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, dataset, sensitive):
        raise Exception("Cannot make predictions for an empty model")
