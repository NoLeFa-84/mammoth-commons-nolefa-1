from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from mammoth_commons.externals import pd_read_csv


@loader(namespace="mammotheu", version="v053", python="3.13")
def data_csv_rankings(path: str = "", delimiter: str = "|") -> CSV:
    """
    This is a Loader to load .csv files with information about researchers
    Args:
        path: Url or path relative to your locally running instance (e.g.: *./data/researchers/Top&#95;researchers.csv*)
        delimiter: Should match the separator of your CSV file columns (e.g.: '|')
    """
    try:
        raw_data = pd_read_csv(path, on_bad_lines="skip", delimiter=delimiter)
    except:
        raise ValueError(
            "Unable to read the given file.  Please double-check the parameters"
        )

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
