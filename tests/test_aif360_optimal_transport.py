import os
from mammoth_commons import testing

from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.metrics.optimal_transport import optimal_transport


def test_optimal_transport():
    with testing.Env(data_custom_csv, model_onnx, optimal_transport) as env:
        dataset = env.data_custom_csv(
            path="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv",
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
            path="file://localhost//" + os.path.abspath("./data/model.onnx"),
            trained_with_sensitive=True,
        )
        markdown_result = env.optimal_transport(dataset, model, sensitive)
        markdown_result.show()


if __name__ == "__main__":
    test_optimal_transport()
