import importlib

from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader, Options
from collections import OrderedDict


@loader(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=("pandas", "ucimlrepo"),
)
def data_uci(
    dataset_name: Options("Credit", "Bank", "Adult", "KDD") = None,
) -> CSV:
    """
    <img src="https://storage.googleapis.com/kaggle-datasets-images/2417096/4083793/85e682cbc981e5214668824a9b0415c3/dataset-cover.jpg?t=2022-08-17-05-14-29" alt="Based on UCI" style="float: left; margin-right: 5px; margin-top: 10px; margin-bottom: 5px; height: 80px;"/>

    Loads a dataset from the UCI Machine Learning Repository (<a href="https://archive.ics.uci.edu/ml/index.php" target="_blank">www.uci.org</a>) containing numeric, categorical, and predictive data columns. The dataset is automatically downloaded from the repository, and basic preprocessing is applied to identify the column types. The specified target column is treated as the predictive label.
        To customize the loading process (e.g., use a different target column, load a subset of features, or handle missing data differently), additional parameters or a custom loader can be used.

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
