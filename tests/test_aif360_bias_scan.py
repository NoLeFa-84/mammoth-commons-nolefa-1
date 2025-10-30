import os
from mammoth_commons import testing

from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.metrics.bias_scan import bias_scan


def test_bias_scan():
    with testing.Env(data_custom_csv, model_onnx, bias_scan) as env:
        dataset = env.data_custom_csv(
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv",
            categorical=[
                "job",
                "marital",
                "education",
                "default",
                "housing",
                "loan",
                "contact",
                "poutcome",
            ],
            numeric=["age", "duration", "campaign", "pdays", "previous"],
            label="y",
            delimiter=";",
        )
        model = env.model_onnx(
            "file://localhost//" + os.path.abspath("./data/model.onnx")
        )
        markdown_result = env.bias_scan(dataset, model, sensitive=[], penalty=0.5)
        markdown_result.show()


if __name__ == "__main__":
    test_bias_scan()
