from mammoth_commons.models import TrivialPredictor
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v053", python="3.13", packages=())
def model_trivial_predictor() -> TrivialPredictor:
    """Creates a trivial predictor that returns the most common predictive label value among provided data.
    If the label is numeric, the median is computed instead. This model servers as an informed baseline
    of what happens even for an uninformed predictor. Several kinds of class biases may exist, for example
    due to different class imbalances for each sensitive attribute dimension (e.g., for old white men
    compared to young hispanic women)."""
    return TrivialPredictor()
