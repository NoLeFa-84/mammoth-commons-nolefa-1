from mammoth_commons.models import EmptyModel
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v054", python="3.13")
def no_model() -> EmptyModel:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/donut.png?raw=true" alt="dataset focus" style="float: left; margin-right: 15px; height: 36px;"/>

    <h3>focus on dataset bias</h3>
    Signifies that the analysis should focus solely on the bias/fairness of the dataset. Different means are used
    to verify the latter. Also consider alternate models that can help analyze dataset biases, like the <i>trivial
    predictor</i>. Not auditing datasets from early on in system creation may irrevocably embed their biases in
    the dataflow in ways that are hard to catch, quantify, or mitigate later.
    """

    return EmptyModel()
