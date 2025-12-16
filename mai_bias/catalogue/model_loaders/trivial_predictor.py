from mammoth_commons.models import TrivialPredictor
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v054", python="3.13", packages=())
def model_trivial_predictor() -> TrivialPredictor:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/nofeats.png?raw=true"
    alt="bias focus" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>uncover biases when features are ignored</h3>
    This is a deliberately biased predictor that ignores dataset features and decides on a fixed prediction based on
    the majority.

    <details><summary><i>How does this work?</i></summary>
    Creates a trivial predictor that returns the most common predictive label value among provided data.
    If the label is numeric, the median is computed instead. This model servers as an informed baseline
    of what happens even for an uninformed predictor. Several kinds of class biases may exist, for example
    due to different class imbalances for each sensitive attribute dimension (e.g., for old white men
    compared to young hispanic women).</details>"""
    return TrivialPredictor()
