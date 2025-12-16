from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from typing import List, Optional
from mammoth_commons.externals import pd_read_csv


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("pandas",),
)
def data_custom_csv(
    path: str = "",
    delimiter: str = ",",
    numeric: Optional[
        List[str]
    ] = None,  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: Optional[
        List[str]
    ] = None,  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    label: Optional[str] = None,
    skip_invalid_lines: bool = True,
) -> CSV:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/csv.png?raw=true"
    alt="csv" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>tabular data with custom formatting</h3>

    Uses <a href="https://pandas.pydata.org/">pandas</a> to load
    a CSV file that contains custom specification of numeric, categorical, and predictive data columns.
    Each row corresponds to a different data sample, with the first one sometimes holding column names
    (this is automatically detected).

    Args:
        path: The local file path or a web URL of the file.
        numeric: A list of comma-separated column names that hold numeric data.
        categorical: A list of comma-separated column names that hold categorical data.
        label: The name of the categorical column that holds predictive label for each data sample.
        delimiter: Which character to split loaded csv rows with.
        skip_invalid_lines: Whether to skip invalid lines being read instead of creating an error.
    """
    assert path.endswith(".csv"), "A file or url with the .csv extension is expected."
    if isinstance(categorical, str):
        categorical = [cat.strip() for cat in categorical.split(",")]
    if isinstance(numeric, str):
        numeric = [num.strip() for num in numeric.split(",")]
    raw_data = pd_read_csv(
        path,
        on_bad_lines="skip" if skip_invalid_lines else "error",
        delimiter=delimiter,
    )
    assert (
        raw_data.shape[1] != 1
    ), "Only one column was found. This often indicates that the wrong delimiter was specified."
    assert label in raw_data, (
        f"The dataset has no column name `{label}` to set as a label. "
        f"\nAvailable columns are {', '.join(raw_data.columns)}"
    )
    for col in categorical:
        assert col in raw_data, (
            f"The dataset has no column name `{col}` to add to categorical attributes. "
            f"Available column are {', '.join(raw_data.columns)}"
        )
    for col in numeric:
        assert col in raw_data, (
            f"The dataset has no column name `{col}` to add to numerical attributes. "
            f"Available columns are {', '.join(raw_data.columns)}"
        )
    csv_dataset = CSV(
        raw_data,
        num=numeric,
        cat=categorical,
        labels=label,
    )
    return csv_dataset
