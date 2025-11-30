# Author: Swati Swati (swati17293@gmail.com, swati.swati@unibw.de)
"""
Fairness Visualization Report Generator
This module generates a detailed fairness plots using the Fairlearn library, providing both group-wise and scalar fairness metrics across sensitive attributes such as sex or race.
"""

from mammoth_commons.datasets import Dataset, Labels
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
def viz_fairness_plots(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
) -> HTML:
    """
    <img src="https://raw.githubusercontent.com/fairlearn/fairlearn/29f6d6f67eea061ae5dae72e976f2069cb38772e/docs/static_landing_page/images/fairlearn_logo.svg" alt="Based on FairLearn" style="float: left; margin-right: 15px; height: 36px;"/>

    <h3>structured visualization of common types of bias</h3>

    <p>
    This module visualizes fairness metrics using the <a href="https://fairlearn.org/" target="_blank">Fairlearn</a> library and interactive Plotly charts.
    It provides visual insights into how a model performs across different groups defined by sensitive features such as gender, race, or age.
    </p>

    <details><summary><i>What to expect?</i></summary>
        <p>
        Two sets of visual outputs are processed:
        </p>
        <ul>
            <li><strong>Group-wise metrics</strong>: Shown as grouped bar charts, these display performance metrics (e.g., false positive rate) across subgroups.</li>
            <li><strong>Scalar metrics</strong>: Displayed as horizontal bar charts, these summarize disparities (e.g., equal opportunity difference) in a compact, interpretable format.</li>
        </ul>
        <p>
            Interactive charts allow users to hover for precise values, compare metrics between groups, and quickly identify fairness gaps.
            An explanation panel is included to define each metric and guide interpretation.
            This module is well suited for exploratory analysis, presentations, and fairness monitoring.
            It makes group disparities visible and intuitive, helping identify where further scrutiny or mitigation may be needed.
        </p>
    </details>
    """
    from mmm_fair_cli.fairlearn_report import generate_reports_from_fairlearn

    # 1. Unwrap model if needed
    if hasattr(model, "mmm"):
        model = model.mmm

    # 2. Predictions
    y_pred = model.predict(dataset, sensitive)

    # 3. Get label column from CSV version
    dataset = dataset.to_csv(sensitive)
    y_true = list(dataset.labels.columns.values())[-1]

    # 4. Raw sensitive values (not encoded)
    sa_df = dataset.df[sensitive].copy()
    raw_sa = sa_df.to_numpy()

    # 5. Group label mappings
    group_mappings = {}
    for attr in sensitive:
        vals = sa_df[attr].unique().tolist()
        group_mappings[attr] = {val: i for i, val in enumerate(vals)}

    # 6. Generate HTML-based Plotly report
    html_string = generate_reports_from_fairlearn(
        report_type="html",  # ← this uses the Plotly charts
        sensitives=sensitive,
        mmm_classifier=model,
        saIndex_test=raw_sa,  # ← pass raw strings
        y_pred=y_pred,
        y_test=y_true,
        launch_browser=False,
        group_mappings=group_mappings,
    )

    # 7. Optional: responsive layout injection (less needed for Plotly but still safe)
    html_string = html_string.replace(
        "<body>", '<body style="margin: 0; padding: 20px; box-sizing: border-box;">'
    )

    return HTML(html_string)
