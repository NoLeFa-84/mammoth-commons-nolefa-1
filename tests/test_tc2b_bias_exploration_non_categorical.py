import os

from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.metrics.interactive_report import interactive_report
from mai_bias.catalogue.metrics.croissant import croissant


def test_bias_exploration():
    with testing.Env(data_custom_csv, model_onnx, interactive_report, croissant) as env:
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
        env.interactive_report(dataset, model, sensitive=["marital", "age"]).show()
        env.croissant(
            dataset,
            None,
            sensitive=["marital"],
            name="Bank dataset with marital and age sensitive attributes",
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            description="The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.",
            citation="A data-driven approach to predict the success of bank telemarketing, By Sérgio Moro, P. Cortez, P. Rita. 2014, Published in Decision Support Systems",
            qualitative_creators="Sérgio Moro, P. Cortez, P. Rita",
            distribution="https://archive.ics.uci.edu/dataset/222/bank+marketing",
        ).show("docs/validation/bank_2_sensitive.html")


if __name__ == "__main__":
    test_bias_exploration()
