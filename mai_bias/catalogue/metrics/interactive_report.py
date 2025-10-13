from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import Dict, List
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories


@metric(
    namespace="mammotheu",
    version="v0049",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def interactive_report(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
) -> HTML:
    """<img src="https://fairbench.readthedocs.io/fairbench.png" alt="Based on FairBench" style="float: left; margin-right: 5px; margin-bottom: 5px; width: 80px;"/>

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
