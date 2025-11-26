from mammoth_commons.datasets.dataset import Dataset


class Text(Dataset):
    def __init__(self, text: str):
        super().__init__(None)
        self.text = text
