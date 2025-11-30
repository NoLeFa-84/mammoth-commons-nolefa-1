# Author: Swati Swati (swati17293@gmail.com, swati.swati@unibw.de)
"""
Fairness Visualization Report Generator
This module generates a detailed fairness report using the Fairlearn library, providing both group-wise and scalar fairness metrics across sensitive attributes such as sex or race.
"""

from mammoth_commons.datasets import Dataset
from mammoth_commons.exports import HTML
from mammoth_commons.models import Predictor
from mammoth_commons.integration import metric

from typing import List


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=(
        "fairlearn",
        "plotly",
        "pandas",
        "onnxruntime",
        "mmm-fair-cli",
        "skl2onnx",
    ),
)
def viz_fairness_report(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
) -> HTML:
    """
    <img src="https://raw.githubusercontent.com/fairlearn/fairlearn/29f6d6f67eea061ae5dae72e976f2069cb38772e/docs/static_landing_page/images/fairlearn_logo.svg" alt="Based on FairLearn" style="float: left; margin-right: 15px; height:36px;"/>

    <h3>structured report on common types of bias</h3>

    <p>
        This module generates a structured fairness report using the <a href="https://fairlearn.org/" target="_blank">Fairlearn</a> library.
        It assesses whether a machine learning model behaves similarly across different population groups, as defined by sensitive attributes such as gender, race, or age.
    </p>

    <details><summary><i>What to expect?</i></summary>
    <p>
        Two types of fairness metrics are considered:
    </p>
    <ul>
        <li><strong>Group-wise metrics</strong>: These show how the model performs for each group separately (e.g., true positive rates for Group A vs. Group B).</li>
        <li><strong>Scalar metrics</strong>: These summarize disparities across groups into single numeric values. Small differences and ratios close to 1 indicate balanced treatment.</li>
    </ul>
    <p>
        Results are presented in aligned tables with clear formatting, allowing users to compare outcomes across groups at a glance.
        Each metric is briefly explained to help interpret whether the model exhibits performance or outcome disparities for different groups.
        This module is particularly useful in evaluation pipelines, audit reports, and model reviews where transparency and fairness are essential.
        It helps teams assess group-level equity in model behavior using interpretable, tabular summaries.
    </p>
    </details>
    """
    from mmm_fair_cli.fairlearn_report import generate_reports_from_fairlearn
    import numpy as np

    # Unwrap model if needed
    if hasattr(model, "mmm"):
        model = model.mmm

    # Predictions
    y_pred = model.predict(dataset, sensitive)

    # Export to CSV style and get true label column name
    dataset = dataset.to_csv(sensitive)
    y_true = list(dataset.labels.columns.values())[-1]

    # Extract sensitive attributes as raw values
    sa_df = dataset.df[sensitive].copy()
    raw_sa = sa_df.to_numpy()

    # Build group mappings for label display
    group_mappings = {}
    for attr in sensitive:
        vals = sa_df[attr].unique().tolist()
        group_mappings[attr] = {val: i for i, val in enumerate(vals)}

    # Generate scrollable, wrapped HTML report from updated CLI
    html_string = generate_reports_from_fairlearn(
        report_type="table",
        sensitives=sensitive,
        mmm_classifier=model,
        saIndex_test=raw_sa,  # use raw string values (e.g. "White", "Black")
        y_pred=y_pred,
        y_test=y_true,
        launch_browser=False,
        group_mappings=group_mappings,  # show real group names
    )

    # Inject better layout into the HTML itself
    # Robust layout replacement to ensure fully responsive side-by-side display
    html_string = html_string.replace(
        '<div class="container">',
        """
        <div class="container"
            style="
                display: flex;
                flex-direction: row;
                flex-wrap: nowrap;
                width: 100%;
                height: auto;
                gap: 30px;
                align-items: flex-start;
                box-sizing: border-box;
            ">
        """,
    )

    html_string = html_string.replace(
        '<div class="report"',
        """
        <div class="report"
            style="
                flex: 3 1 0;
                min-width: 0;
                overflow-x: auto;
                overflow-y: visible;
                white-space: nowrap;
                box-sizing: border-box;
            "
        """,
    )

    html_string = html_string.replace(
        '<div class="explanation"',
        """
        <div class="explanation"
            style="
                flex: 1 1 0;
                min-width: 0;
                word-break: break-word;
                overflow-wrap: anywhere;
                box-sizing: border-box;
            "
        """,
    )

    # Optional: Patch <body> if needed to remove padding/margin conflicts
    html_string = html_string.replace(
        "<body>", '<body style="margin: 0; padding: 20px; box-sizing: border-box;">'
    )

    return HTML(html_string)
