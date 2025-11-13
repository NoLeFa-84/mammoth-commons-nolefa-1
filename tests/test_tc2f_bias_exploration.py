import os
from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.metrics.aif360_metrics import aif360_metrics


def test_bias_exploration():
    with testing.Env(data_custom_csv, model_onnx, aif360_metrics) as env:
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

        # age, duration, campaign, pdays, previous
        # job, marital, education, default, housing, loan, contact, poutcome
        # y ;

        model_path = "file://localhost//" + os.path.abspath("./data/model.onnx")
        print(
            model_path
        )  # e.g., file://localhost///home/maniospas/Documents/mammoth-commons/data/model.onnx
        model = env.model_onnx(model_path, trained_with_sensitive=True)

        markdown_result = env.aif360_metrics(dataset, model, sensitive)
        markdown_result.show()


if __name__ == "__main__":
    test_bias_exploration()
