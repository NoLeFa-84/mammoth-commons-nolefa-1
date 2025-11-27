from typing import List
from mammoth_commons.datasets import Dataset, CSV


class ImageLike(Dataset):
    def to_csv(self, sensitive: list[str], skip_first_columns: int = 1):
        from mammoth_commons.externals import pd_read_csv
        import pandas as pd

        def is_numeric(c):
            if not pd.api.types.is_any_real_numeric_dtype(c):
                return False
            return len(set(c)) > 10

        df = pd_read_csv(self.path)
        cols = [col for col in df][skip_first_columns:]
        numeric = [col for col in cols if is_numeric(df[col])]
        num = [col for col in numeric if len(set(df[col])) > 10]
        num_set = set(numeric)
        cat = [col for col in cols if col not in num_set if col != self.target]
        label = self.target
        csv = CSV(df, num=num, cat=cat, labels=label)
        csv.description = self.description.replace(
            "Args:",
            "<br><br><i>Fairness analysis only accounts for sensitive information encoded in the tabular segment of the dataset.</i>\nArgs:",
        )
        return csv


class Image(ImageLike):
    def __init__(
        self,
        path,
        root_dir,
        target,
        data_transform,
        batch_size,
        shuffle,
        num_workers,
        cols,
    ):
        """
        Args:
            path (str): Path to the CSV file with annotations (should involve the columns path|attribute1|...|attributeN).
            root_dir (str): Root image dataset directory.
            target (str): The target attribute to be predicted.
            data_transform (callable): A function/transform that takes in an image and returns a transformed version.
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Set to True to have the data reshuffled every time they are obtained.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__(None)
        target = str(target)
        cols = [str(col) for col in cols]

        assert target in cols, f"Target {target} not one of the columns: " + ",".join(
            cols
        )
        self.path = path
        self.root_dir = root_dir
        self.target = target
        self.data_transform = data_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cols = cols
        self.input_size = self._get_input_size(self.data_transform)
        self.num_workers = num_workers

    def to_torch(self, sensitive: List[str]):
        # dynamic dependencies here to not force a torch dependency on commons from components that don't need it
        from torch.utils.data import DataLoader
        from mammoth_commons.datasets.backend.torch_implementations import (
            PytorchImageDataset,
        )
        import os
        import warnings

        if os.name == "nt":  # Windows
            if self.num_workers != 0:
                warnings.warn(
                    "Multi-worker data loading is not supported on Windows. "
                    "Setting num_workers=0."
                )
            self.num_workers = 0
        torch_dataset = PytorchImageDataset(
            csv_path=self.path,
            root_dir=self.root_dir,
            target=self.target,
            sensitive=sensitive,
            data_transform=self.data_transform,
        )

        return DataLoader(
            dataset=torch_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def to_numpy(self, sensitive: List[str]):
        from mammoth_commons.datasets.backend.onnx_transforms import torch2onnx
        from mammoth_commons.datasets.backend.onnx_implementations import (
            ONNXImageDataset,
            numpy_dataloader_image,
        )

        onnx_transforms = torch2onnx(self.data_transform)
        dataset = ONNXImageDataset(
            csv_path=self.path,
            root_dir=self.root_dir,
            target=self.target,
            sensitive=sensitive,
            data_transform=onnx_transforms,
        )

        return numpy_dataloader_image(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def _get_input_size(self, transform):
        from torchvision import transforms

        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size
        return (224, 224)
