import os
from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.metrics.specific_concerns import specific_concerns


def test_bias_exploration():
    with testing.Env(data_custom_csv, model_onnx, specific_concerns) as env:
        numeric = ["age", "duration", "campaign", "pdays", "previous"]
        categorical = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "poutcome",
        ]
        sensitive = ["marital"]
        dataset_uri = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
        dataset = env.data_custom_csv(
            dataset_uri,
            categorical=categorical,
            numeric=numeric,
            label="y",
            delimiter=";",
        )

        model_path = "file://localhost//" + os.path.abspath("./data/model.onnx")
        model = env.model_onnx(model_path, trained_with_sensitive=True)
        env.specific_concerns(
            dataset, model, sensitive, base_measure="Accuracy", reduction="Min"
        ).show()
        env.specific_concerns(
            dataset,
            model,
            sensitive="marital,age",
            base_measure="Accuracy",
            reduction="Min",
            intersections="Subgroups",
        ).show()


if __name__ == "__main__":
    test_bias_exploration()
