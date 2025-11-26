import numpy as np
from mammoth_commons.models.predictor import Predictor


class ONNX(Predictor):
    def __init__(self, model_bytes, includes_sensitive=False):
        super().__init__()
        self.model_bytes = model_bytes
        self.includes_sensitive = includes_sensitive

    def predict(self, dataset, sensitive: list[str]):
        includes_sensitive = (
            not self.includes_sensitive
        )  # TODO: investigate to_pred if this matches its semnatics in the following line
        x = (
            dataset
            if isinstance(dataset, np.ndarray)
            else dataset.to_pred(sensitive if includes_sensitive else list())
        )
        import onnxruntime as rt
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

        sess = rt.InferenceSession(self.model_bytes, providers=["CPUExecutionProvider"])
        onnx_type_to_np = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        np_type = onnx_type_to_np.get(sess.get_inputs()[0].type, None)
        if not np_type:
            raise Exception(
                "Onnx model has been saved to expect and unknown format: "
                + sess.get_inputs()[0].type
            )
        x = x.astype(np_type)
        assert (
            len(sess.get_inputs()[0].shape) == 2
        ), "Onnx model has been saved to expect a non-2D matrix"
        assert (
            sess.get_inputs()[0].shape[1] == x.shape[1]
        ), f"Onnx model has been saved to expect {sess.get_inputs()[0].shape[1]} input columns but you provided a dataset with {x.shape[1]} columns. Maybe you included/excluded some attributes, like sensitive ones?"
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        try:
            return sess.run([label_name], {input_name: x})[0]
        except InvalidArgument as e:
            raise Exception(
                "The ONNx loader's runtime encountered an error that typically occurs "
                "when the selected dataset is incompatible to the loaded model. "
                "Consult with the model provider whether your are loading the "
                "model properly. If you are investigating a dataset, "
                "consider switching to trained-on-the-fly model loaders.<br><br>"
                '<details><summary class="btn btn-secondary">Details</summary><br><br>'
                "<pre>" + str(e) + "</pre></details>"
            )
