from mammoth_commons.models import ONNX, ONNXEnsemble
from mammoth_commons.integration import loader
from mammoth_commons.externals import prepare
import re
import numpy as np
import zipfile


@loader(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=("onnxruntime", "mmm-fair-cli", "skl2onnx"),
)
def model_onnx_ensemble(path: str = "") -> ONNXEnsemble:
    """
    <img src="https://raw.githubusercontent.com/arjunroyihrpa/MMM_fair/main/images/mmm-fair.png" alt="Based on MMM-Fair" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 80px;"/>

    <p>This ONNX Ensemble Module enables predictions using a <a href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank">boosting ensemble</a> mechanism, ideal for combining multiple weak learners to improve prediction accuracy. Boosting, a powerful technique in machine learning, focuses on training a series of simple models (weak learners) – often single-depth <a href="https://scikit-learn.org/stable/modules/tree.html#classification" target="_blank">decision trees</a> – and combining them into a strong ensemble model. However, the model loader module allows any model converted to <a href="https://onnxruntime.ai/docs/tutorials/traditional-ml.html#convert-model-to-onnx">ONNX</a> format and zipped inside a directory path along with other meta-informations (if any) stored in <a href="https://numpy.org/doc/2.1/reference/generated/numpy.save.html">.npy</a> format.</p>
    <p><b>Usage Instructions:</b> To load a model, users need to supply a zip file path. This zip file should include at least one or possibly many trained models, each saved in the ONNX format, as well as parameters, such as weights (often denoted as ‘alphas’), that define each learner’s contribution to the final model. For an example of preparing this file, please see <a href="https://github.com/mammoth-eu/mammoth-commons/blob/dev/tests/test-mfppb-onnx-ensemble.ipynb" target="_blank">our notebook</a>.</p>
    <p>The module recomends using the <a href="https://pypi.org/project/mmm-fair/"> MMM-Fair models</a>. MMM-Fair is a fairness-aware machine learning framework designed to support high-stakes AI decision-making under competing fairness and accuracy demands. The three M’s stand for: • Multi-Objective: Optimizes across classification accuracy, balanced accuracy, and fairness (specifically, maximum group-level discrimination). • Multi-Attribute: Supports multiple protected groups (e.g., race, gender, age) simultaneously, analyzing group-specific disparities. • Multi-Definition: Evaluates and compares fairness under multiple definitions—Demographic Parity (DP), Equal Opportunity (EP), and Equalized Odds (EO).
    MMM-Fair enables developers, researchers, and decision-makers to explore the full spectrum of possible trade-offs and select the model configuration that aligns with their social or organizational goals. For theoretical understanding of MMM-Fair, it is recomended to read the <a href="https://link.springer.com/chapter/10.1007/978-3-031-18840-4_21" target="_blank">published scientific article</a> that introduced the foundation of the MMM-algorithms.</p>
    <p><b>Train and upload: </b> To create and integrate your own MMM-Fair model trained on your intended data, please follow the instructions given in the <a href="https://pypi.org/project/mmm-fair/" target="_blank">Pypi package guidance</a>.</p>

    Args:
        path: A zip file containing the ensemble elements such as learners/models, weight parameters, etc.
    """

    models = []
    model_names = []
    params = None

    def myk(name):
        return int(re.findall(r"[+-]?\d+", name)[0])

    # Read the zip file
    with zipfile.ZipFile(prepare(path)) as myzip:
        # Extract and load the weights file
        for file_name in myzip.namelist():
            if file_name.endswith(".npy"):
                with myzip.open(file_name) as param_file:
                    params = np.load(param_file, allow_pickle=True)

            elif file_name.endswith(".onnx"):
                model_names.append(file_name)

        if len(model_names) > 1:
            model_names.sort(key=myk)

        for file_name in model_names:
            with myzip.open(file_name) as model_file:
                model_bytes = model_file.read()
                models.append(model_bytes)
    params_dict = dict(params.item()) if params is not None else {"-": None}
    return ONNXEnsemble(models, **params_dict)
