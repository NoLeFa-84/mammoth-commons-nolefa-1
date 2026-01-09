from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from mammoth_commons.externals import pd_read_csv


@loader(namespace="mammotheu", version="v054", python="3.13")
def data_csv_rankings(path: str = "", delimiter: str = "|") -> CSV:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/csv.png?raw=true"
    alt="csv" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>anonymized researcher characteristics</h3>

    Uses <a href="https://pandas.pydata.org/">pandas</a> to load
    CSV file with information about researcher citations, productivity, gender, nationality,
    country region, and income.

    Args:
        path: Url or path relative to your locally running instance (e.g.: *./data/researchers/Top&#95;researchers.csv*)
        delimiter: Should match the separator of your CSV file columns. Default is '|'.
    """
    raw_data = pd_read_csv(path, on_bad_lines="skip", delimiter=delimiter)
    validate_input(raw_data)
    csv_dataset = CSV(
        raw_data,
        num=["Citations", "Productivity"],
        cat=[
            "Nationality",
            "Nationality_Region",
            "Nationality_IncomeGroup",
            "aff_country",
            "aff_country_Region",
            "aff_country_IncomeGroup",
            "Gender",
        ],
        labels=[
            "id"
        ],  # Just a dummy right now.  We don't do supervised learning and don't "label" anything
    )
    return csv_dataset


def validate_input(data):
    required_columns = [
        "Citations",
        "Productivity",
        "Nationality_Region",
        "Nationality_IncomeGroup",
        "Gender",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns must be present in the dataset, but they are not: {missing_columns}"
        )
