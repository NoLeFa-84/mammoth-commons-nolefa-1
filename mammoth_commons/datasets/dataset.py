from typing import Any


class Labels:
    def __init__(self, columns: dict[str, Any], name: str = "class"):
        assert isinstance(columns, dict), "Labels should be a dict"
        self.columns = {str(k): v for k, v in columns.items()}
        self.name = name

    def __getitem__(self, item: str):
        return self.columns[item]

    def __iter__(self):
        return self.columns.__iter__()

    def __len__(self):
        return len(self.columns)

    def items(self):
        return self.columns.items()


class Dataset:
    integration = "dsl.Dataset"

    def __init__(self, labels: Labels | None):
        self.labels = labels
        self.description: str | dict | None = None

    def to_numpy(self, features: list[str] | None = None):
        raise Exception(
            f"Dataset {self.__class__.__name__} has no numpy conversion for features"
        )

    def to_pred(self, exclude: list[str]):
        raise Exception(
            f"Dataset {self.__class__.__name__} has no numpy conversion given excluded features"
        )

    def to_csv(self, sensitive: list[str]):
        raise Exception(f"Dataset {self.__class__.__name__} cannot be treated as a csv")

    def to_description(self):
        if not self.description:
            return ""
        if isinstance(self.description, str):
            desc = self.description.split("Args:")[0] + "<br>"
        elif isinstance(self.description, dict):
            desc = ""
            for key, value in self.description.items():
                desc += f"<h3>{key}</h3>" + value.replace("\n", "<br>") + "<br>"
        else:
            raise Exception("Dataset description must be a string or a dictionary.")
        from mammoth_commons.exports import get_description_header

        header = get_description_header(desc)
        return header + desc.replace(header, "", 1)
