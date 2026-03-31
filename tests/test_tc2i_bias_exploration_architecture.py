from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.auto_csv import data_auto_csv
from mai_bias.catalogue.model_loaders.popular_architecture import model_from_sklearn
from mai_bias.catalogue.metrics.model_card import model_card
from mai_bias.catalogue.metrics.croissant import croissant


def test_bias_exploration():
    with testing.Env(data_auto_csv, model_from_sklearn, model_card, croissant) as env:
        dataset = env.data_auto_csv("data/bank.csv")
        model = env.model_from_sklearn()
        env.model_card(dataset, model, sensitive=["marital"]).show()


if __name__ == "__main__":
    test_bias_exploration()
