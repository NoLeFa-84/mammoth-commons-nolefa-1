import importlib

from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from collections import OrderedDict
from typing import Literal


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("pandas", "ucimlrepo"),
)
def data_uci(
    dataset_name: Literal["Credit", "Bank", "Adult", "KDD"] = None,
) -> CSV:
    """
    <img src="https://storage.googleapis.com/kaggle-datasets-images/2417096/4083793/85e682cbc981e5214668824a9b0415c3/dataset-cover.jpg?t=2022-08-17-05-14-29"
    alt="UCI" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>tabular datasets from UCI</h3>
    Loads a dataset from the (<a href="https://archive.ics.uci.edu/ml/index.php" target="_blank">UCI</a>) machine
    learning dataset repository. The dataset contains pre-specified numeric, categorical, and predictive data columns,
    as well as preprocessing. Available datasets are commonly used in the algorithmic fairness literature to test
    new approaches.

    Args:
        dataset_name: The name of the dataset.
    """
    pd = importlib.import_module("pandas")
    uci = importlib.import_module("ucimlrepo")
    name = dataset_name.lower()
    target = {"credit": "Y", "bank": "y", "adult": "income", "kdd": "income"}[name]
    repo_id = {"credit": 350, "bank": 222, "adult": 2, "kdd": 117}[name]
    repo = uci.fetch_ucirepo(id=repo_id)
    # label = repo.data.features[target].copy()
    # repo.data.features = repo.data.features.drop(columns=[target])
    label = repo.data.targets[target]
    num = [
        col
        for col in repo.data.features
        if pd.api.types.is_any_real_numeric_dtype(repo.data.features[col])
        and len(set(repo.data.features[col])) > 10
        and col != target
    ]
    num_set = set(num)
    cat = [col for col in repo.data.features if col not in num_set and col != target]
    csv_dataset = CSV(
        repo.data.features,
        num=num,
        cat=cat,
        labels=label,
    )
    csv_dataset.description = OrderedDict(
        [
            ("Summary", repo.metadata.additional_info.summary),
            ("Variables", repo.metadata.additional_info.variable_info),
        ]
    )
    return csv_dataset
