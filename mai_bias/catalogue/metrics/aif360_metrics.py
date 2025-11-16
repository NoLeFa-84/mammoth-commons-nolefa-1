import math

from mammoth_commons.datasets import Dataset, Labels
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric
from mammoth_commons.externals import align_predictions
from mammoth_commons.integration_callback import notify_progress, notify_end
import json


def render_metric_bars(rows, sensitive):
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
            chart_data.append(
                {
                    "metric": metric,
                    "group": attr,
                    "value": val,
                }
            )
    data_json = json.dumps(chart_data)

    def render_constant_table():
        if not constant_metrics:
            return ""
        rows = "".join(
            f"<tr><td>{entry['Metric']}</td><td><code>{round(entry['Value'], 3):.3f}</code></td></tr>"
            for entry in constant_metrics
        )
        return f"""
        <h3>Overall performance</h3>
        <table class="table table-bordered table-sm">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    d3_script = f"""
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
    return render_constant_table() + d3_script


@metric(
    namespace="mammotheu",
    version="v053",
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
) -> HTML:
    """
    <img src="https://ai-fairness-360.org/" alt="Based on AIF360" style="float: left; margin-right: 5px; margin-bottom: 5px; width: 80px;"/>
    <p>This module evaluates fairness using IBM's <a href="https://aif360.readthedocs.io" target="_blank">AIF360</a> library.
    It computes standard group fairness metrics for each sensitive attribute provided. If attributes are non-binary, they are
    binarized into one-hot encoded columns. Only categorical attributes are allowed.</p>

    <span class="alert alert-warning alert-dismissible fade show" role="alert"
        style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i>
        This module is based on AIF360, which does not support generalized intersectional analysis.
        When needed, generate sensitive intersectional group labels in your dataset. However, these
        will always be computed against the rest of the population.</span>

    Args:
        favorable_label: The prediction label value which is considered favorable (i.e. "positive"). Default is 1 for binary classifiers.
        unfavorable_label: The prediction label value which is considered unfavorable (i.e. "negative"). Default is 0 for binary classifiers.
    """

    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    import numpy as np

    assert len(sensitive) > 0, "You must specify at least one sensitive attribute"
    if isinstance(sensitive, str):
        sensitive = [sens.strip() for sens in sensitive.split(",")]
    original_sensitive_len = len(sensitive)

    y_pred = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    y_pred, y_true = align_predictions(y_pred, dataset.labels)
    label_col = list(y_true.columns)[0]
    pred_col = list(y_pred.columns)[0]

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

    metrics_by_group = {}
    all_metric_names = set()
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
    prog = 0
    for attr in sensitive:
        privileged = [{attr: 1}]
        unprivileged = [{attr: 0}]

        # bmetric = BinaryLabelDatasetMetric(aif_dataset_true, unprivileged, privileged)
        metric_obj = ClassificationMetric(
            aif_dataset_true, aif_dataset_pred, unprivileged, privileged
        )
        metrics = dict()
        for label, method in classification_metrics.items():
            notify_progress(
                float(prog) / len(classification_metrics) / len(sensitive),
                f"Analyzing attribute {attr} under metric {label}",
            )
            prog += 1
            try:
                value = getattr(metric_obj, method)()
                if isinstance(value, (int, float, np.number)):
                    metrics[label] = abs(float(value))
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    metrics[label] = abs(float(value))
                else:
                    metrics[label] = math.nan
            except Exception:
                metrics[label] = math.nan
        metrics_by_group[attr] = metrics
        all_metric_names.update(metrics_by_group[attr].keys())
    notify_end()
    all_metric_names = sorted(all_metric_names)
    rows = []
    for metric in all_metric_names:
        row = {"Metric": metric}
        for attr in sensitive:
            row[attr] = metrics_by_group.get(attr, {}).get(metric, math.nan)
        rows.append(row)

    # html = f"""
    # <h1>AIF360 Fairness Report</h1>
    # <p>This report shows metric distributions across sensitive groups.</p>
    # <p>See <a href="https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html" target="_blank">AIF360 metric documentation</a> for metric definitions.</p>
    #
    # <details><summary>In total {len(sensitive)} protected groups were analysed. </summary><i>{', '.join(sensitive).replace('_', ' ')}</i><br></details>
    # <details><summary>Metrics</summary><i>{'<table class="table table-sm"><tr><th>Function</th><th>Report Name</th></tr>' + ''.join(f'<tr><td>{method.replace('_', ' ')}</td><td>{label}</td></tr>' for label, method in classification_metrics.items()) + '</table>'}</i><br></details>
    # {'<p class="text-warning"><i>Some sensitive attributes that were not binary have been automatically expanded via one-hot encoding.</i></p>' if original_sensitive_len != len(sensitive) else ''}
    #
    # {render_metric_bars(rows, sensitive)}
    #
    # <div class="mt-4">{dataset.to_description()}</div>
    # """

    # Build the metrics table separately
    metrics_table = (
        '<table class="table table-sm"><tr><th>Function</th><th>Report Name</th></tr>'
        + "".join(
            f"<tr><td>{method.replace('_', ' ')}</td><td>{label}</td></tr>"
            for label, method in classification_metrics.items()
        )
        + "</table>"
    )

    faq_style = """
        <style>
        .faq-container {
          max-width: 600px;
          margin: 20px auto;
          font-family: Arial, sans-serif;
        }
        
        .faq-box {
          border: 1px solid #ccc;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 16px;
          box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
          background: #fff;
        }
        
        .faq-box h3 {
          margin-top: 0;
          font-size: 1.2em;
          color: #333;
        }
        
        .faq-box p {
          margin: 0;
          color: #555;
        }
        </style>
    """

    html = f"""
    <div class="container">
    <h1>AIF360 Fairness Report</h1>
    {faq_style}
    <hr/>
    <div class="faq-container">
        <div class="faq-box">
              <h3>❓ What is this?</h3>
              <p>This is a fairness report compiled with a MAI-BIAS module using the AIF360 library. 
              Results correspond to specific dataset and model loaders and parameters.</p>
              <br/>
              <p>You can see various metrics, grouped into those that assess the overall 
              model, and those that are computed for each protected group, each corresponding to
              a sensitive attribute value (see summary). Metric values should ideally be 
              similar across groups for models to be considered fair. Do not neglect performance, 
              and, after looking at everything, focus only on equalizing measures 
              that matter for your application context
              - it is impossible to optimize for everything.</p>
        </div>
        <div class="faq-box">
              <h3>❗ Summary</h3>
                <p>This report shows metric distributions across sensitive groups. See
                <a href="https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html" 
                target="_blank">AIF360 metric documentation</a> for metric definitions. Some fairness
                metrics are computed for each group by comparing it with the rest of the population.</p>
                <br/>
                <details>
                  <summary>In total {len(sensitive)} protected groups were analysed.</summary>
                  <i>{', '.join(sensitive).replace('_', ' ')}</i><br>
                </details>
                <details>
                  <summary>Computed metrics</summary>
                  <i>{metrics_table}</i><br>
                </details>
                {('<p class="text-warning"><br><i>Some sensitive attributes that were not '
                  'binary have been automatically expanded to their one-hot encoding.'
                  '</i></p>'if original_sensitive_len != len(sensitive) else '')}
                <br>
                <p><b>Results require manual inspection to identify problematic values or imbalances.</b></p>
        </div>
    </div>
    <hr/>
    {render_metric_bars(rows, sensitive)}
    <div class="mt-4">{dataset.to_description()}</div>
    </div>
    """
    return HTML(html)
