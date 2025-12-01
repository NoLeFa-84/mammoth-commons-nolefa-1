import os
from mammoth_commons.models.pytorch import Pytorch
from mammoth_commons.integration import loader
from mammoth_commons.externals import safeexec, prepare


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("numpy", "torch", "torchvision"),
)
def model_torch(
    state_path: str = "",
    model_path: str = "",
    model_name: str = "model",
    safe_libraries: str = "numpy, torch, torchvision, PIL, io, requests",
    multiclass_threshold: float = 0,
) -> Pytorch:
    """
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png"
    alt="PyTorch" style="float: left; margin-right: 5px; height: 32px;"/>
    <h3>deep learning model (for GPU)</h3>

    Loads a <a href="https://pytorch.org/">PyTorch</a> deep learning model that comprises a Python code initializing the
    architecture, and a file of trained parameters.

    Args:
        state_path: The path in which the architecture's state is stored.
        model_path: The path in which the architecture's initialization script resides. Alternatively, you may also just paste the initialization code in this field.
        model_name: The variable in the model path's script to which the architecture is assigned.
        safe_libraries: A comma-separated list of libraries that can be imported. For safety, the architecture's definition is allowed to directly import only specified libraries.
        multiclass_threshold: A decision threshold that treats outputs as separate classes. If this is set to zero (default), a softmax is applied to outputs. For binary classification, this is equivalent to setting the decision threshold at 0.5. Otherwise, each output is thresholded separately.
    """
    import torch

    state_path = prepare(state_path)
    model_path = prepare(model_path)

    multiclass_threshold = float(multiclass_threshold)
    model = safeexec(
        model_path,
        out=model_name,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(state_path, map_location=device))
    return Pytorch(model, threshold=multiclass_threshold)
