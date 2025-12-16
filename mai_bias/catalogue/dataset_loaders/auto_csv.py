import importlib
from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from mammoth_commons.externals import pd_read_csv


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("pandas",),
)
def data_auto_csv(path: str = "", max_discrete: int = 10) -> CSV:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/csv.png?raw=true"
    alt="csv" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>tabular data with common formatting</h3>

    Uses <a href="https://pandas.pydata.org/">pandas</a> to load
    a CSV file that contains numeric, categorical, and predictive data columns.
    This automatically detects the characteristics of the dataset being loaded,
    namely the delimiter that separates the columns, and whether each column contains
    numeric or categorical data.
    The last categorical column is used as the dataset label. To load the file maintaining
    more control over options (e.g., a subset of columns, a different label column) use the
    custom csv loader instead.

    <details><summary><i>How to replicate this data loader during AI creation?</i></summary>
    If you want to train a model while using the same loading mechanism as this dataset,
    run the following Python script. This uses supporting methods from the lightweight
    mammoth-commons core to retrieve <a href="https://numpy.org/">numpy</a>
    arrays *X,y* holding dataset features and categorical labels respectively.

    <small>
    <pre>
    % pip install --upgrade pandas
    % pip install --upgrade mammoth_commons
    import pandas as pd
    from mammoth_commons.externals import pd_read_csv
    from mammoth_commons.datasets import CSV

    # set parameters and load data (modify max_discrete as needed)
    path = ...
    max_discrete = 10
    df = pd_read_csv(path, on_bad_lines="skip")

    # identify numeric and categorical columns
    num = [col for col in df if pd.api.types.is_any_real_numeric_dtype(df[col])]
    num = [col for col in num if len(set(df[col])) > max_discrete]
    num_set = set(num)
    cat = [col for col in df if col not in num_set]

    # convert to numpy data
    csv_dataset = CSV(df, num=num, cat=cat[:-1], labels=cat[-1])
    X = X.astype(np.float32)
    y = df[cat[-1]]
    </pre>
    </small>
    </details>

    Args:
        path: The local file path or a web URL of the file.
        max_discrete: If a numeric column has a number of discrete entries than is less than this number (e.g., if it contains binary numeric values) then it is considered to hold categorical instead of numeric data. Minimum accepted value is 2.
    """
    max_discrete = int(max_discrete)
    assert path.endswith(".csv"), "A file or url with the .csv extension is expected."
    assert max_discrete >= 2, "Numeric levels (max discrete) should be at least 2"
    pd = importlib.import_module("pandas")
    df = pd_read_csv(path, on_bad_lines="skip")
    num = [col for col in df if pd.api.types.is_any_real_numeric_dtype(df[col])]
    num = [col for col in num if len(set(df[col])) > max_discrete]
    num_set = set(num)
    cat = [col for col in df if col not in num_set]
    assert len(cat) >= 1, "At least one categorical column is required."
    csv_dataset = CSV(df, num=num, cat=cat[:-1], labels=cat[-1])
    return csv_dataset
