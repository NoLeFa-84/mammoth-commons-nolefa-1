from mammoth_commons.datasets import CSV
from mammoth_commons.externals import prepare
from mammoth_commons.integration import loader, Options


def data_local(raw_data, target: str = None) -> CSV:
    import pandas as pd

    numeric = [
        col
        for col in raw_data
        if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
        and len(set(raw_data[col])) > 10
    ]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least one categorical column is required.")
    if not target:
        if "class" in categorical:
            target = "class"
        elif "Class" in categorical:
            target = "Class"
        elif "Y" in categorical:
            target = "Y"
        elif "y" in categorical:
            target = "y"
        else:
            target = categorical[-1]
    label = raw_data[target].copy()
    if target in categorical:
        categorical.remove(target)
    elif target in numeric:
        numeric.remove(target)
    raw_data = raw_data.drop(columns=[target])
    return CSV(raw_data, num=numeric, cat=categorical, labels=label)


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("pandas",),
)
def data_read_any(
    dataset_path: str = None,
    target: str = None,
) -> CSV:
    """
    <img src="https://github.com/arjunroyihrpa/MMM_fair/blob/main/images/mmm-fair.png?raw=true"
    alt="MMM-Fair" style="float: left; margin-right: 5px; height: 36px;"/>
    <h3>flexible tabular data loading</h3>

    Uses <a href="https://pandas.pydata.org/">pandas</a> to load file stored in one of the formats:
    `.csv`, `.xls`, `.xlsx`, `.xlsm`, `.xlsb`, `.odf`, `.ods`, `.json`, `.html`, or `.htm`.
    The module is derived from  <a href="https://github.com/arjunroyihrpa/MMM_fair">MMM-fair</a>
    to support several data formats. Basic preprocessing is applied
    to infer column types, and the specified target column is treated as the predictive label.

    Args:
        dataset_path: Path or URL to the dataset file. Must have one of the supported extensions.
        target: The name of the column to treat as the predictive label.
    """
    import csv
    import string
    import pandas as pd

    dataset_path = prepare(dataset_path)
    try:
        if dataset_path.endswith(".csv"):
            try:
                with open(dataset_path, "r") as file:
                    sample = file.read(1024)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    delimiter = str(delimiter)
                    if delimiter in string.ascii_letters:
                        common_delims = [",", ";", "|", "\t"]
                        counts = {d: sample.count(d) for d in common_delims}
                        # pick the one with highest count, fallback to ","
                        delimiter = (
                            max(counts, key=counts.get) if any(counts.values()) else ","
                        )
            except Exception:
                delimiter = None
            df = pd.read_csv(dataset_path, delimiter=delimiter)
        elif dataset_path.endswith(".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods"):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        elif dataset_path.endswith(".html", ".htm"):
            df = pd.read_html(dataset_path)
        return data_local(df, target)
    except Exception as e:
        raise ValueError(
            f"Could not read data. Unsupported or invalid format for {dataset_path}"
        )
