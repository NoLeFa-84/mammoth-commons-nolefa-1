# Author: Swati Swati (swati17293@gmail.com, swati.swati@unibw.de)
"""
Fairness Visualization Plots Generator
This module generates interactive fairness plots using the Fairlearn library, providing both group-wise and scalar fairness metrics across sensitive attributes such as sex or race.
"""

from mammoth_commons.datasets import Dataset
from mammoth_commons.exports import HTML, simplified_formatter
from mammoth_commons.models import Predictor
from mammoth_commons.integration import metric
from mammoth_commons.reminders import on_results

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
    problematic_deviation: float = 0.2,
) -> HTML:
    """
    <img src="https://raw.githubusercontent.com/fairlearn/fairlearn/29f6d6f67eea061ae5dae72e976f2069cb38772e/docs/static_landing_page/images/fairlearn_logo.svg" alt="Based on FairLearn" style="float: left; margin-right: 15px; height:36px;"/>

    <h3>structured visualization of common types of bias</h3>

    <p>
        This module visualizes fairness metrics using the <a href="https://fairlearn.org/" target="_blank">Fairlearn</a> library and interactive Plotly charts.
        It provides visual insights into how a model performs across different groups defined by sensitive features such as gender, race, or age.
    </p>

    <details><summary><i>What to expect?</i></summary>
    <p>
        Two sets of visual outputs are produced:
    </p>
    <ul>
        <li><strong>Group-wise metrics</strong>: Grouped bar charts displaying performance metrics (e.g., false positive rate) across subgroups.</li>
        <li><strong>Scalar metrics</strong>: Horizontal bar charts summarizing disparities (e.g., equal opportunity difference) in a compact, interpretable format.</li>
    </ul>
    <p>
        Interactive charts allow users to hover for precise values, compare metrics between groups, and quickly identify fairness gaps.
        This module is well suited for exploratory analysis, presentations, and fairness monitoring.
    </p>
    </details>

    Args:
        problematic_deviation: Threshold for flagging bias. Difference metrics (e.g. demographic_parity_difference) above this value, or ratio metrics (e.g. demographic_parity_ratio) falling below 0.8, are counted as problematic. Default is 0.2.
    """
    from mmm_fair_cli.fairlearn_report import generate_reports_from_fairlearn
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equal_opportunity_difference,
        equal_opportunity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
    )

    prob = float(problematic_deviation)
    assert 0 <= prob <= 1, "problematic_deviation must be in [0, 1]"

    dataset_description = dataset.to_description()
    model_description = model.to_description()

    y_pred = model.predict(dataset, sensitive)

    dataset = dataset.to_csv(sensitive)
    y_true = list(dataset.labels.columns.values())[-1]

    if hasattr(model, "mmm"):
        model = model.mmm

    sa_df = dataset.df[sensitive].copy()
    raw_sa = sa_df.to_numpy()

    group_mappings = {}
    for attr in sensitive:
        vals = sa_df[attr].unique().tolist()
        group_mappings[attr] = {val: i for i, val in enumerate(vals)}

    per_attr_flags: dict[str, list[str]] = {}

    for i, attr in enumerate(sensitive):
        sensitive_column = raw_sa[:, i]

        scalars = {
            "demographic_parity_difference": demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
            "demographic_parity_ratio": demographic_parity_ratio(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
            "equal_opportunity_difference": equal_opportunity_difference(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
            "equal_opportunity_ratio": equal_opportunity_ratio(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
            "equalized_odds_difference": equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
            "equalized_odds_ratio": equalized_odds_ratio(
                y_true, y_pred, sensitive_features=sensitive_column
            ),
        }

        flags = []
        for name, value in scalars.items():
            if "_ratio" in name:
                if value < (1.0 - prob):
                    flags.append(name)
            else:
                if value > prob:
                    flags.append(name)
        per_attr_flags[attr] = flags

    total_flagged = sum(len(f) for f in per_attr_flags.values())
    outcome = "fair" if total_flagged == 0 else "biased"
    title_text = (
        f"Biases in {total_flagged} benefit(s)"
        if total_flagged > 0
        else "no fairness concerns"
    )

    # --- Build per-attribute status summary table for the "about" panel ---
    sensitive_list_html = ", ".join(f"<i>{s}</i>" for s in sensitive)

    attr_status_rows = ""
    for attr, flags in per_attr_flags.items():
        n = len(flags)
        if n == 0:
            detail = "No metrics flagged. The model treats all subgroups within this attribute consistently."
        else:
            flagged_names = ", ".join(f"<i>{f}</i>" for f in flags)
            detail = f"{n} metric(s) flagged: {flagged_names}."

        attr_status_rows += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 6px 10px; vertical-align:top;"><b>{attr}</b></td>
                <td style="padding: 6px 10px; vertical-align:top;">{detail}</td>
            </tr>"""

    about_html = f"""
        <p>This report visualizes whether the model behaves similarly across different population groups,
        as defined by the following sensitive attributes: {sensitive_list_html}</p>
        <p>Two types of fairness metrics are visualized:
        <ul>
            <li><strong>Group-wise metrics</strong>: Per-group performance values shown as grouped bar charts.</li>
            <li><strong>Scalar metrics</strong>: Single-value summaries of disparity shown as horizontal bar charts (differences close to 0 and ratios close to 1 indicate balanced treatment).</li>
        </ul></p>
        <p>A metric is flagged as biased if its difference exceeds <b>{prob:.1f}</b> or its ratio falls below <b>{1.0 - prob:.1f}</b>.</p>
        <table style="width:100%; border-collapse: collapse; font-size: 13px; margin-top: 10px;">
            <thead>
                <tr style="background:#f0f4ff;">
                    <th style="padding:6px 10px; text-align:left;">Attribute</th>
                    <th style="padding:6px 10px; text-align:left;">Detail</th>
                </tr>
            </thead>
            <tbody>{attr_status_rows}</tbody>
        </table>
    """

    # --- Generate the Plotly HTML charts (goes into experts panel) ---
    html_string = generate_reports_from_fairlearn(
        report_type="html",
        sensitives=sensitive,
        mmm_classifier=model,
        saIndex_test=raw_sa,
        y_pred=y_pred,
        y_test=y_true,
        launch_browser=False,
        group_mappings=group_mappings,
    )

    import re

    # --- STEP 1: Strip inline styles from .report and .explanation divs ---
    html_string = re.sub(
        r'(<div[^>]*class="report"[^>]*?)style="[^"]*"',
        r'\1',
        html_string
    )
    html_string = re.sub(
        r'(<div[^>]*class="explanation"[^>]*?)style="[^"]*"',
        r'\1',
        html_string
    )

    # --- STEP 2: Fix ALL divs with width: 50% — makes inner panels full width ---
    html_string = re.sub(
        r'(style="[^"]*?)width\s*:\s*50%([^"]*")',
        r'\1width: 100%\2',
        html_string
    )

    # --- STEP 3: Fix bias threshold text ---
    html_string = html_string.replace(
        "<p>Generally, difference &gt; <strong>0.1</strong> or ratio &lt; <strong>0.8</strong> may indicate bias.</p>",
        f"<p>Generally, difference &gt; <strong>{prob:.1f}</strong> or ratio &lt; <strong>{1.0 - prob:.1f}</strong> may indicate bias.</p>"
    )

    # --- STEP 4: Patch hardcoded values inside the embedded <style> block ---
    html_string = html_string.replace("margin: 40px;", "margin: 10px;")
    html_string = html_string.replace("gap: 40px;", "gap: 0px;")
    html_string = html_string.replace("max-width: 500px;", "max-width: none;")
    html_string = html_string.replace("max-width: 460px;", "max-width: none;")

    # --- STEP 5: Inject scoped override styles and wrap in unique id ---
    style_block = """<style>
#mmm-expert-panel * { box-sizing: border-box !important; }
#mmm-expert-panel body { margin: 10px !important; }
#mmm-expert-panel .container {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    width: 95% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    gap: 20px !important;
    align-items: flex-start !important;
}
#mmm-expert-panel .report {
    flex: 0 0 55% !important;
    max-width: 55% !important;
    width: 55% !important;
    min-width: 0 !important;
    padding: 10px !important;
    overflow-x: auto !important;
    white-space: normal !important;
}
#mmm-expert-panel .explanation {
    flex: 0 0 40% !important;
    max-width: 40% !important;
    width: 40% !important;
    min-width: 0 !important;
    padding: 20px !important;
    background-color: #fcfcfc !important;
    border: 1px solid #eee !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    white-space: normal !important;
    margin-left: 0 !important;
}
#mmm-expert-panel .explanation div {
    width: 100% !important;
    max-width: 100% !important;
}
</style>"""

    html_string = '<div id="mmm-expert-panel">' + style_block + html_string + '</div>'

    return HTML(
        simplified_formatter(
            outcome=outcome,
            title=title_text,
            technology='<div><img src="https://raw.githubusercontent.com/fairlearn/fairlearn/29f6d6f67eea061ae5dae72e976f2069cb38772e/docs/static_landing_page/images/fairlearn_logo.svg" alt="logo" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 48px;"/> <h1>based on Fairlearn reporting</h1></div>',
            about=about_html,
            methodology=f"""
                <p>The report was generated using the <a href="https://fairlearn.org/" target="_blank">Fairlearn</a> library.
                For each sensitive attribute, group membership is determined by the unique values present in the dataset.
                Group-wise metrics are computed independently per group and displayed as interactive Plotly charts.
                Scalar metrics aggregate these into summary statistics that capture the degree of disparity across groups.</p>
                <p>A metric is considered biased if the gap between groups is too large. Specifically, difference metrics are flagged when they exceed <b>{prob:.1f}</b> (closer to 0 is fairer), and ratio metrics are flagged when they drop below <b>{1.0 - prob:.1f}</b> (closer to 1 is fairer).</p>
                <p>The analysis considered <b>{len(sensitive)}</b> sensitive attribute(s):
                <br>{sensitive_list_html}</p>
            """,
            pipeline=f"{dataset_description}<br><br>{model_description}",
            experts=html_string,
        )
    )