from mammoth_commons.models import ONNX
from mammoth_commons.integration import loader
from mammoth_commons.externals import prepare


@loader(namespace="mammotheu", version="v054", python="3.13", packages=("onnxruntime",))
def model_onnx(path: str = "", trained_with_sensitive: bool = True) -> ONNX:
    """
    <img src="https://onnx.ai/images/ONNX-Logo.svg"
    alt="ONNX" style="background-color: #000055; float: left; margin-right: 15px; height: 32px;"/>
    <h3>a serialized machine learning model</h3>

    Loads an inference model stored in the <a href="https://onnx.ai/">ONNx</a> format,
    which is a generic cross-platform way of representing AI models with a common set of operations.
    The loaded model should be compatible with the dataset being analysed, for example having been trained on
    the same tabular data columns.

    <details><summary><i>Technical details and how to export a model to this format.</i></summary>
    Several machine learning frameworks can export to ONNx. The latter
    supports several different runtimes, but this module's implementation selects
    the `CPUExecutionProvider` runtime to run on to maintain compatibility
    with most machines.
    For inference in GPUs, prefer storing and loading models in formats
    that are guaranteed to maintain all features that could be included in the architectures
    of respective frameworks; this can be achieved with other model loaders.

    Here are some quick links on how to export ONNx models from popular frameworks:
    <ul>
    <li><a href="https://onnx.ai/sklearn-onnx">scikit-learn</a></li>
    <li><a href="https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html">PyTorch</a></li>
    <li><a href="https://onnxruntime.ai/docs/tutorials/tf-get-started.html">TensorFlow</a></li>
    </ul>
    </details>

    Args:
        path: A local path or url pointing to the ONNX file. The loader checks for the existence of the local path, and if it does not exist downloads it locally from the provided URL before loading.
        trained_with_sensitive: Whether model training included the sensitive attributes that will be analysed in the next step or not. Including those attributes could help mitigate bias for some bias-aware training algorithms. Leave checked if you just trained the model with all available attributes.
    """
    with open(prepare(path), "rb") as f:
        model_bytes = f.read()
    return ONNX(model_bytes, trained_with_sensitive)
