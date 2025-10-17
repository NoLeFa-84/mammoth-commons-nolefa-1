import io
import os
from mammoth_commons.models.pytorch2onnx import ONNX
from mammoth_commons.integration import loader
from mammoth_commons.externals import safeexec
import tempfile


@loader(
    namespace="mammotheu",
    version="v0049",
    python="3.13",
    packages=("numpy", "torch", "torchvision", "onnxscript"),
)
def model_torch2onnx(
    state_path: str = "",
    model_path: str = "",
    model_name: str = "model",
    input_width: int = 224,
    input_height: int = 224,
    safe_libraries: str = "numpy, torch, torchvision, PIL, io, requests",
    multiclass_threshold: float = 0,
) -> ONNX:
    """

    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="Based on PyTorch" style="float: left; margin-right: 5px; margin-bottom: 5px; margin-top: 10px; height: 30px;"/>

    Loads a <a href="https://pytorch.org/">pytorch</a> model that comprises a Python code initializing the
    architecture and a file of trained parameters, and converts into ONNX format to support processing by
    modules not supporting GPU compute. For safety, the architecture's
    definition is allowed to directly import only specified libraries.

    Args:
        state_path: The path in which the architecture's state is stored.
        model_path: The path in which the architecture's initialization script resides. Alternatively, you may also just paste the initialization code in this field.
        model_name: The variable in the model path's script to which the architecture is assigned.
        input_width: The expected width of input images.
        input_height: The expected heightg of input images.
        safe_libraries: A comma-separated list of libraries that can be imported.
        multiclass_threshold: A decision threshold that treats outputs as separate classes. If this is set to zero (default), a softmax is applied to outputs. For binary classification, this is equivalent to setting the decision threshold at 0.5. Otherwise, each output is thresholded separately.
    """
    import torch

    input_width = int(input_width)
    input_height = int(input_height)

    multiclass_threshold = float(multiclass_threshold)
    model = safeexec(
        model_path,
        out=model_name,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )

    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    dummy_input = torch.randn(1, 3, input_width, input_height)

    # TODO: temporary monkey patch of normalization
    import torchvision.transforms._functional_tensor as F_t

    def safe_normalize(tensor, mean, std, inplace=False):
        # TorchVision-compatible signature, but no (std == 0).any() guard
        dtype = tensor.dtype
        device = tensor.device
        mean = torch.as_tensor(mean, dtype=dtype, device=device).view(1, -1, 1, 1)
        std = torch.as_tensor(std, dtype=dtype, device=device).view(1, -1, 1, 1)
        if inplace:
            tensor.sub_(mean).div_(std)
            return tensor
        return (tensor - mean) / std

    F_t.normalize = safe_normalize

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
        onnx_model_path = temp_file.name
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    onnx_model = ONNX(onnx_model_path, threshold=multiclass_threshold)
    os.remove(onnx_model_path)
    return onnx_model
