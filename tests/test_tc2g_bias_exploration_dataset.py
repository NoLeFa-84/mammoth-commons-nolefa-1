from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.uci_csv import data_uci
from mai_bias.catalogue.model_loaders.no_model import no_model
from mai_bias.catalogue.metrics.sklearn_visual_analysis import sklearn_visual_analysis
from mai_bias.catalogue.metrics.croissant import croissant


def test_bias_exploration():
    with testing.Env(data_uci, no_model, sklearn_visual_analysis, croissant) as env:
        dataset = env.data_uci("credit")
        model = env.no_model()
        env.sklearn_visual_analysis(dataset, model, sensitive=["X2", "X4"]).show()
        env.croissant(
            dataset,
            None,
            sensitive=["marital"],
            name="Credit dataset with sex (X2) and marital status (X4) sensitive attributes",
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            description="This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.",
            citation="The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients, By I. Yeh, Che-hui Lien. 2009, Published in Expert systems with applications",
            qualitative_creators="I. Yeh, Che-hui Lien",
            distribution="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        ).show("docs/validation/credit_2_sensitive.html")


if __name__ == "__main__":
    test_bias_exploration()
