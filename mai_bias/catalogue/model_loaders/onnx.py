from mammoth_commons.models import ONNX
from mammoth_commons.integration import loader
from mammoth_commons.externals import prepare


@loader(namespace="mammotheu", version="v053", python="3.13", packages=("onnxruntime",))
def model_onnx(path: str = "", trained_with_sensitive: bool = True) -> ONNX:
    """
    <img src="https://onnx.ai/images/ONNX-Logo.svg" alt="Based on ONNX" style="background-color: #000055; float: left; margin-right: 15px; margin-top: 5px; margin-bottom: 5px; height: 40px;"/>

    Loads an inference model stored in <a href="https://onnx.ai/">ONNx</a> format.
    This is a generic cross-platform format for representing machine learning models with a common set of operations.
    Several machine learning frameworks can export to this format.
    The loaded model should be compatible with the dataset being analysed. For example,
    the same data columns as in the dataset should be used for training on tabular data.

    ONNx supports several different runtimes, but this loader's implementation selects
    the `CPUExecutionProvider` runtime to run on, therefore maintaining compatibility
    with most machines.
    For inference in GPUs, prefer storing and loading models in formats
    that are guaranteed to maintain all features that could be included in the architectures
    of respective frameworks; this can be achieved with different model loaders.

    Here are some quick links on how to export ONNx models from popular frameworks:
    <ul>
    <li><a href="https://onnx.ai/sklearn-onnx">scikit-learn</a></li>
    <li><a href="https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html">PyTorch</a></li>
    <li><a href="https://onnxruntime.ai/docs/tutorials/tf-get-started.html">TensorFlow</a></li>
    </ul>

    Args:
        path: A local path or url pointing to the ONNX file. The loader checks for the existence of the local path, and if it does not exist downloads it locally from the provided URL before loading.
        trained_with_sensitive: Whether model training included the sensitive attributes that will be analysed in the next step or not. Including those attributes could help mitigate bias for some bias-aware training algorithms. Leave checked if you just trained the model with all available attributes.
    """
    with open(prepare(path), "rb") as f:
        model_bytes = f.read()
    return ONNX(model_bytes, trained_with_sensitive)
