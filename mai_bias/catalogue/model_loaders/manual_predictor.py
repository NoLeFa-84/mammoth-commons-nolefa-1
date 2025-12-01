from mammoth_commons.models import Predictor
from mammoth_commons.integration import loader
from mammoth_commons.externals import pd_read_csv, fb_categories


class ManualPredictor(Predictor):
    def __init__(self, predictions: list):
        super().__init__()
        self.predictions = fb_categories(predictions)

    def predict(self, dataset, sensitive: list[str]):
        return self.predictions


@loader(namespace="mammotheu", version="v054", python="3.13", packages=("pandas",))
def model_manual_predictor(path_or_predictions: str = "") -> ManualPredictor:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/list.png?raw=true"
    alt="dataset focus" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>manual predictions for the dataset</h3>

    Manually loaded predictions may have been produced by workflows external to the toolkit.

    <details><summary>How to format the predictions.</summary>
    Input comma-separated list of predictions that correspond to the data you are processing.
    This is useful so that you can export the predictions directly from your testing code. If there are no
    commas provided, this module's argument is considered to be the URL to a CSV file whose last column contains
    the predictions. Other columns are ignored, but are allowed to give you flexibility. If your dataset also
    has the last column as the prediction label (e.g., if it is loaded with auto csv) you will obtain analysis
    for a perfect predictor.
    </details>

    Args:
            path_or_predictions: A comma-separated list of predictions, or a URL to a CSV file whose last column contains them.
    """
    if isinstance(path_or_predictions, str) and "," not in path_or_predictions:
        df = pd_read_csv(path_or_predictions)
        path_or_predictions = df.iloc[:, -1].tolist()
    else:
        path_or_predictions = [pred.strip() for pred in path_or_predictions.split(",")]
    return ManualPredictor(path_or_predictions)
