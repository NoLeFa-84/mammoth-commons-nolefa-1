from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import Dict, List, Literal
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def interactive_report(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Literal["Pairwise", "To the total population"] = "Pairwise",
) -> HTML:
    """<img src="https://github.com/mever-team/FairBench/blob/main/docs/fairbench.png?raw=true" alt="Based on FairBench"
    style="float: left; margin-right: 5px; margin-bottom: 5px; width: 36px;"/>
    <h3>for data scientists: explore several biases and their intermediate quantities</h3>

    Creates an interactive report using the FairBench library. The report creates traceable evaluations that
    you can shift through to find actual sources of unfairness.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population.
    """
    from fairbench import v1 as fb

    # obtain predictions
    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)

    # declare sensitive attributes
    labels = dataset.labels
    predictions, labels = align_predictions(predictions, dataset.labels)
    predictions = predictions.columns
    sensitive = fb.Fork(
        {attr + " ": fb_categories(dataset.df[attr]) for attr in sensitive}
    )

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.multireport if compare_groups == "Pairwise" else fb.unireport
    if labels is None:
        report = report_type(predictions=predictions, sensitive=sensitive)
    else:
        report = fb.Fork(
            {
                label
                + " ": report_type(
                    predictions=predictions,
                    labels=(
                        labels[label].to_numpy()
                        if hasattr(labels[label], "to_numpy")
                        else labels[label]
                    ),
                    sensitive=sensitive,
                )
                for label in labels
            }
        )

    return HTML(
        "<div class='container'><h1>Interactive report</h1>\n"
        + fb.interactive_html(report, show=False, name="Classes")
        + "</div>"
    )
