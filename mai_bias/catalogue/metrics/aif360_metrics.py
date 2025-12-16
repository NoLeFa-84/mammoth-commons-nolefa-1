import math
from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric
from mammoth_commons.externals import align_predictions
from mammoth_commons.integration_callback import notify_progress, notify_end
from mammoth_commons.reminders import on_results
import json


def render_metric_bars(rows, sensitive):
    # (unchanged original function)
    constant_metrics = []
    varying_metrics = []
    for row in rows:
        metric = row["Metric"]
        values = [row.get(attr) for attr in sensitive]
        numeric_values = [
            v
            for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        if len(numeric_values) > 0 and all(
            v == numeric_values[0] for v in numeric_values
        ):
            constant_metrics.append({"Metric": metric, "Value": numeric_values[0]})
        else:
            varying_metrics.append(row)

    chart_data = []
    for row in varying_metrics:
        metric = row["Metric"]
        for attr in sensitive:
            val = row.get(attr)
            try:
                val = float(val)
            except:
                val = None
            chart_data.append({"metric": metric, "group": attr, "value": val})

    data_json = json.dumps(chart_data)

    def render_constant_table():
        if not constant_metrics:
            return ""
        rows = "".join(
            f"<tr><td>{entry['Metric']}</td>"
            f"<td><code>{round(entry['Value'], 3):.3f}</code></td></tr>"
            for entry in constant_metrics
        )
        return f"""
        <h3>Overall performance</h3>
        <table class="table table-bordered table-sm">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    return (
        render_constant_table()
        + f"""
<div id="chart-container"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {{
    const data = {data_json};
    const grouped = d3.group(data.filter(d => d.value !== null), d => d.metric);
    const container = d3.select("#chart-container");

    grouped.forEach((values, metric) => {{
        container.append("h3").text(metric);

        const width = 600;
        const barHeight = 25;
        const margin = {{ top: 10, right: 70, bottom: 10, left: 150 }};
        const actualMax = d3.max(values, d => Math.abs(d.value));
        const maxVal = actualMax <= 1 ? 1 : actualMax;

        const x = d3.scaleLinear()
            .domain([0, maxVal])
            .range([0, width - margin.left - margin.right]);

        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", values.length * (barHeight + 10) + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

        const color = d3.scaleLinear()
            .domain([0, maxVal])
            .range(["#f88", "#4caf50"]);

        g.selectAll("rect")
            .data(values)
            .join("rect")
            .attr("x", 0)
            .attr("y", (d, i) => i * (barHeight + 10))
            .attr("width", d => x(d.value))
            .attr("height", barHeight)
            .attr("fill", d => color(d.value));

        g.selectAll("text.label")
            .data(values)
            .join("text")
            .attr("class", "label")
            .attr("x", -10)
            .attr("y", (d, i) => i * (barHeight + 10) + barHeight / 2 + 4)
            .attr("text-anchor", "end")
            .text(d => d.group.replace("_", " "));

        g.selectAll("text.value")
            .data(values)
            .join("text")
            .attr("class", "value")
            .attr("x", d => x(d.value) + 5)
            .attr("y", (d, i) => i * (barHeight + 10) + barHeight / 2 + 4)
            .text(d => d.value.toFixed(3));
    }});
}})();
</script>
"""
    )


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=(
        "aif360",
        "pandas",
        "scikit-learn",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def aif360_metrics(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    favorable_label: int = 1,
    unfavorable_label: int = 0,
    bias_threshold: float = 0.05,
) -> HTML:
    """
    <img src="https://avatars.githubusercontent.com/u/56103733?s=48&v=4"
    alt="Based on AIF360" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 36px;"/>

    <h3>popular types of bias</h3>

    <p>Use IBM's <a href="https://aif360.readthedocs.io" target="_blank">AIF360</a> to compute
    common group fairness metrics for each sensitive attribute provided. If attributes are non-binary, they are
    binarized into one-hot encoded columns. Only categorical attributes are allowed.</p>

    <span class="alert alert-warning alert-dismissible fade show" role="alert"
        style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i>
        This module is based on AIF360, which does not support generalized intersectional analysis.
        When needed, generate sensitive intersectional group labels in your dataset. However, these
        will always be computed against the rest of the population.</span>

    Args:
        favorable_label: The prediction label value which is considered favorable (i.e. "positive"). Default is 1 for binary classifiers.
        unfavorable_label: The prediction label value which is considered unfavorable (i.e. "negative"). Default is 0 for binary classifiers.
        bias_threshold: The maximum value of bias assessment. Common literature default is 0.1 or 0.2, but stakeholder engagement in the MAMMOth project suggests that this value should be lower for measures of bias that matter. Hence a more conservative default of 0.05 is used.
    """

    from aif360.metrics import ClassificationMetric
    import numpy as np

    threshold = float(bias_threshold)
    if isinstance(sensitive, str):
        sensitive = [s.strip() for s in sensitive.split(",")]
    assert len(sensitive) > 0, "Must specify at least one sensitive attribute"
    y_pred = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    y_pred, y_true = align_predictions(y_pred, dataset.labels)
    pred_col = list(y_pred.columns)[0]
    label_col = list(y_true.columns)[0]
    dataset.df["label"] = y_true[label_col]
    aif_dataset_true, sensitive = dataset.to_aif360(
        label_col="label",
        sensitive_cols=sensitive,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )
    df = dataset.df.copy()
    df["y_pred"] = y_pred[pred_col]
    aif_dataset_pred = aif_dataset_true.copy()
    aif_dataset_pred.labels = df[["y_pred"]].values

    classification_metrics = {
        "Accuracy": "accuracy",
        "Average Abs Odds Difference": "average_abs_odds_difference",
        "Average Odds Difference": "average_odds_difference",
        "Average Predictive Value Difference": "average_predictive_value_difference",
        "Base Rate": "base_rate",
        "Between All Groups CV": "between_all_groups_coefficient_of_variation",
        "Between All Groups GEI": "between_all_groups_generalized_entropy_index",
        "Between All Groups Theil Index": "between_all_groups_theil_index",
        "Between Group CV": "between_group_coefficient_of_variation",
        "Between Group GEI": "between_group_generalized_entropy_index",
        "Between Group Theil Index": "between_group_theil_index",
        "Coefficient of Variation": "coefficient_of_variation",
        "Bias Amplification": "differential_fairness_bias_amplification",
        "Disparate Impact": "disparate_impact",
        "Equal Opportunity Difference": "equal_opportunity_difference",
        "Equalized Odds Difference": "equalized_odds_difference",
        "Error Rate": "error_rate",
        "Error Rate Difference": "error_rate_difference",
        "Error Rate Ratio": "error_rate_ratio",
        "False Discovery Rate": "false_discovery_rate",
        "False Discovery Rate Difference": "false_discovery_rate_difference",
        "False Discovery Rate Ratio": "false_discovery_rate_ratio",
        "False Negative Rate": "false_negative_rate",
        "False Negative Rate Difference": "false_negative_rate_difference",
        "False Negative Rate Ratio": "false_negative_rate_ratio",
        "False Omission Rate": "false_omission_rate",
        "False Omission Rate Difference": "false_omission_rate_difference",
        "False Omission Rate Ratio": "false_omission_rate_ratio",
        "False Positive Rate": "false_positive_rate",
        "False Positive Rate Difference": "false_positive_rate_difference",
        "False Positive Rate Ratio": "false_positive_rate_ratio",
        "Gen. Entropy Index": "generalized_entropy_index",
        "Gen. Equalized Odds Difference": "generalized_equalized_odds_difference",
        "Gen. False Negative Rate": "generalized_false_negative_rate",
        "Gen. False Positive Rate": "generalized_false_positive_rate",
        "Gen. True Negative Rate": "generalized_true_negative_rate",
        "Gen. True Positive Rate": "generalized_true_positive_rate",
        "Negative Predictive Value": "negative_predictive_value",
        "Num False Negatives": "num_false_negatives",
        "Num False Positives": "num_false_positives",
        "Num Gen. False Negatives": "num_generalized_false_negatives",
        "Num Gen. False Positives": "num_generalized_false_positives",
        "Num Gen. True Negatives": "num_generalized_true_negatives",
        "Num Gen. True Positives": "num_generalized_true_positives",
        "Num Instances": "num_instances",
        "Num Negatives": "num_negatives",
        "Num Positives": "num_positives",
        "Num Pred. Negatives": "num_pred_negatives",
        "Num Pred. Positives": "num_pred_positives",
        "Num True Negatives": "num_true_negatives",
        "Num True Positives": "num_true_positives",
        "Positive Predictive Value": "positive_predictive_value",
        "Selection Rate": "selection_rate",
        "Smoothed EDF": "smoothed_empirical_differential_fairness",
        "Statistical Parity Difference": "statistical_parity_difference",
        "Theil Index": "theil_index",
        "True Negative Rate": "true_negative_rate",
        "True Positive Rate": "true_positive_rate",
        "True Positive Rate Difference": "true_positive_rate_difference",
    }

    metrics_by_group = {}
    all_metric_names = set()
    prog = 0
    for attr in sensitive:
        privileged = [{attr: 1}]
        unprivileged = [{attr: 0}]
        metric_obj = ClassificationMetric(
            aif_dataset_true, aif_dataset_pred, unprivileged, privileged
        )
        metrics = {}
        for label, method in classification_metrics.items():
            notify_progress(
                float(prog) / len(classification_metrics) / len(sensitive),
                f"Analyzing {attr}: {label}",
            )
            prog += 1
            try:
                v = getattr(metric_obj, method)()
                if isinstance(v, (int, float, np.number)):
                    metrics[label] = abs(float(v))
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    metrics[label] = abs(float(v))
                else:
                    metrics[label] = math.nan
            except Exception:
                metrics[label] = math.nan

        metrics_by_group[attr] = metrics
        all_metric_names.update(metrics.keys())
    notify_end()
    rows = []
    for metric_name in sorted(all_metric_names):
        row = {"Metric": metric_name}
        for attr in sensitive:
            row[attr] = metrics_by_group[attr].get(metric_name, math.nan)
        rows.append(row)
    IDEAL_VALUES = {
        "Disparate Impact": 1.0,
        "Statistical Parity Difference": 0.0,
        "Equal Opportunity Difference": 0.0,
        "Equalized Odds Difference": 0.0,
        "Average Odds Difference": 0.0,
        "Average Abs Odds Difference": 0.0,
        "False Positive Rate Difference": 0.0,
        "False Negative Rate Difference": 0.0,
        "Error Rate Difference": 0.0,
        "False Discovery Rate Ratio": 1.0,
        "False Omission Rate Ratio": 1.0,
        "False Omission Rate Difference": 0.0,
        "False Positive Rate Ratio": 0.0,
    }
    biases = {
        r["Metric"].lower()
        for r in rows
        if r["Metric"] in IDEAL_VALUES
        for attr, v in r.items()
        if attr != "Metric"
        and v is not None
        and not math.isnan(v)
        and abs(IDEAL_VALUES[r["Metric"]] - float(v)) > threshold
    }
    verdict = (
        (
            list(biases)[0][0].upper()
            + list(biases)[0][1:]
            + f" bias"  # " in {len(sensitive)} groups"
        )
        if len(biases) == 1
        else (
            f"{len(biases)} types of bias"  # " in {len(sensitive)} groups"
            if len(biases)
            else f"No concerns"  # " among {len(sensitive)} groups"
        )
    )
    bias_list_html = (
        "<i>" + "<br>".join(biases) + "</i>" if biases else "the system is likely fair"
    )
    html_content = f"""
    <style>
        .pill-buttons {{display: flex; gap: 12px; margin: 20px 0;}}
        .banner {{width: 100%;  padding: 180px 24px; font-size: 64px; font-weight: 700; text-align: center; color: white; border-radius: 12px margin-bottom: 25px;}}
        .banner.fair {{ background: #2e8b57; }}
        .banner.biased {{ background: #c0392b; }}
        .banner.report {{ background: #7f8c8d; }}
        .pill-btn {{ width:100%; text-align:center; padding: 10px 18px; background: #f5f5f5; border-radius: 10px; border: 1px solid #cccccc; cursor: pointer; font-size: 18px; transition: background 0.2s;}}
        .pill-btn:hover {{ background: #e0e0e0; }}
        .pill-btn.active {{ background: #d0d0d0; border-color: #999999;}}
        .section-panel {{ display: none; padding: 12px; border: 0px; }}
        .section-panel.active {{ display: block; }}
        .overview-title {{font-size: 32px; font-weight: 700; margin-top: 0; margin-bottom: 10px; }}
        .overview-sub {{ font-size: 18px; opacity: 0.8; margin-bottom: 20px; }}
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const buttons = document.querySelectorAll(".pill-btn");
            const sections = document.querySelectorAll(".section-panel");
            buttons.forEach(btn => {{
                btn.addEventListener("click", () => {{
                    let t = btn.getAttribute("data-target");
                    buttons.forEach(b => b.classList.remove("active"));
                    sections.forEach(s => s.classList.remove("active"));
                    btn.classList.add("active");
                    document.getElementById(t).classList.add("active");
                }});
            }});
            buttons[0].classList.add("active");
            sections[0].classList.add("active");
        }});
    </script>
    <div>
        <h1 class="banner {'biased' if 'bias' in verdict else 'fair'}">{verdict}</h1>
        <div><img src="https://avatars.githubusercontent.com/u/56103733?s=48&v=4" alt="Based on AIF360" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 48px;"/> <h1>&nbsp;based on AIF360 metrics</h1></div>
        <div class="pill-buttons">
            <div class="pill-btn" data-target="whatis">
                What is this?<br>
                <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/question.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="warning">
                Responsible use
                <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/warning.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="methodology">
                Analysis methodology<br>
                <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/methodology.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="pipeline">
                Data pipeline<br>
                <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/data.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="experts">
                For experts<br>
                <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/chart.png?raw=true" height="128px"/>
            </div>
        </div>
        <div id="whatis" class="section-panel">
            <p>We used IBM’s AIF360 library to checks for common types of bias and found the following:</p>{bias_list_html}
        </div>
        <div id="warning" class="section-panel">
            {on_results}
        </div>
        <div id="methodology" class="section-panel">
            <p>Each fairness metric provided by AIF360 is computed across <b>{len(sensitive)}</b> groups, each of which 
            is compared to the rest of the population. No group intersections are accounted for. 
            We check whether notions of bias exceed <b>{threshold}</b> in a scale 0-1 where 0 represents biased systems, 
            or whether notions of fairness are lesser than <b>{1-threshold}</b> in a scale 0-1 where 1 represents fair systems.</p>
            Considered groups are: <i><br>{"<br>".join(sens.replace('_', ' ') for sens in sensitive)}</i>
        </div>
        <div id="pipeline" class="section-panel">
            {dataset.to_description().split("Args:")[0]}<br><br>{model.to_description().split("Args:")[0]}
        </div>
        <div id="experts" class="section-panel">
            {render_metric_bars(rows, sensitive)}
        </div>
    </div>
    """

    return HTML(html_content)
