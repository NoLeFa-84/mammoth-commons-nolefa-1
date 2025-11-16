from mammoth_commons.models import EmptyModel
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v053", python="3.13")
def no_model() -> EmptyModel:
    """Signifies that the analysis should focus solely on the fairness of the dataset.
    Not treating bias at early steps may irrevocably embed it in the dataflow in ways that are hard to catch and quantify later.
    """

    return EmptyModel()
