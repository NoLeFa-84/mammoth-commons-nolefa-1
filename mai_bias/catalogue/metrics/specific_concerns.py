import importlib

from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from mammoth_commons.reminders import on_results
from typing import List, Literal
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def specific_concerns(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Literal["Base", "All", "Subgroups"] = "Subgroups",
    base_measure: Literal[
        "Accuracy",
        "True positive rate",
        "True negative rate",
        "Area under curve",
        "Positive rate",
    ] = "Accuracy",
    compare_groups: Literal["Pairwise", "To the total population"] = "Pairwise",
    reduction: Literal[
        "Min",
        "Max",
        "Weighted mean",
        "Max relative difference",
        "Max betweeness area",
        "Standard deviation x2",
        "Gini coefficient",
    ] = "Max relative difference",
    problematic_deviation: float = 0.05,
) -> HTML:
    """
    <img src="https://github.com/mever-team/FairBench/blob/main/docs/fairbench.png?raw=true" alt="Based on FairBench"
    style="float: left; margin-right: 5px; margin-bottom: 5px; height: 36px;"/>

    <h3>focus on a specific definition of fairness</h3>

    <p>Computes a fairness or bias measure that matches a specific type of numerical
    evaluation using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. The measure is built by combining simpler options to form more than 300 valid alternatives.</p>

    <span class="alert alert-warning alert-dismissible fade show" role="alert"
    style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i>
    This computes a specific fairness concerns and does not paint a broad enough picture. Make sure that
    you explore prospective biases with other modules first, like <i>model card</i>.</span>

    <details><summary><i>Technical details.</i></summary>

    <p>The assessment is conducted over sensitive attributes like gender, age, and race. Each attribute can have
    multiple values, such as several genders or races. Numeric attributes, like age, are normalized to the range [0,1]
    and treated as fuzzy values, where 0 indicates membership to a fuzzy group of "small" values, and 1 indicates
    membership to a fuzzy group of "large" values. A separate set of fairness metrics is calculated for each prediction
    label.</p>

    <p>If intersectional subgroup analysis is enabled, separate subgroups are created for each combination of sensitive
    attribute values. However, if there are too many attributes, some groups will be small or empty. Empty groups are
    ignored in the analysis.</p>
    </details>

    Args:
        intersections: Whether to consider only the provided groups (Base), all non-empty group intersections (All), or all non-empty intersections while ignoring larger groups during analysis (Subgroups). For example, the last option may not contain a `White` dimension if `White Men` is an existing dimension. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected.
        base_measure: A base measure of algorithmic performance to be computed on each group.
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        reduction: The strategy with which to reduce all measure comparisons to one value.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
    """
    fb = importlib.import_module("fairbench")
    if isinstance(sensitive, str):
        sensitive = sensitive.split(",")
    assert len(sensitive) != 0, "At least one sensitive attribute should be selected"
    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    sensitive = fb.Dimensions(
        {s: fb_categories(dataset.df[s]) for s in sensitive}, _separator=" "
    )
    if intersections != "Base":
        sensitive = sensitive.intersectional(delimiter=" - ")
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    assert (
        len(sensitive.branches()) != 0
    ), "Could not find any sensitive attribute intersections"
    predictions, labels = align_predictions(predictions, dataset.labels)
    predictions = predictions.columns
    labels = labels.columns if labels else None

    fb_measures = {
        "Accuracy": "acc",
        "True positive rate": "tpr",
        "True negative rate": "tnr",
        "Positive rate": "pr",
        "Area under curve": "auc",
    }
    fb_reductions = {
        "Min": "min",
        "Max": "max",
        "Weighted mean": "wmean",
        "Max difference": "maxdiff" if compare_groups != "vsall" else "largestmaxdiff",
        "Max relative difference": (
            "maxrel" if compare_groups != "vsall" else "largestmaxrel"
        ),
        "Max betweeness area": (
            "maxbarea" if compare_groups != "vsall" else "largestmaxbarea"
        ),
        "Standard deviation x2": "stdx2",
        "Gini coefficient": "gini",
    }
    metric_name = (
        ("pairwise" if compare_groups == "Pairwise" else "vsall")
        + "_"
        + fb_reductions[reduction]
        + "_"
        + fb_measures[base_measure]
    )

    report = fb.quick.__getattr__(metric_name)(
        predictions=predictions, labels=labels, sensitive=sensitive
    )
    problematic_deviation = float(problematic_deviation)
    assert (
        0 <= problematic_deviation <= 1
    ), "Problematic deviation should be in the range [0,1]"
    if problematic_deviation:
        report = report.filter(
            fb.investigate.DeviationsOver(problematic_deviation, prune=False)
        )

    full_report = report.show(
        env=fb.export.Html(view=False, filename=None),
        depth=1 if isinstance(predictions, dict) else 0,
    )

    dataset_description = dataset.to_description().split("Args:")[0]
    model_description = model.to_description().split("Args:")[0]
    outcome = (
        "Report"
        if problematic_deviation == 0
        else ("Fair" if report.flatten(True)[0] < problematic_deviation else "Biased")
    ) + f" {base_measure.lower()}"  # " in {len(sensitive.branches())} protected groups"

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
                    let target = btn.getAttribute("data-target");
                    buttons.forEach(b => b.classList.remove("active"));
                    sections.forEach(s => s.classList.remove("active"));
                    btn.classList.add("active");
                    document.getElementById(target).classList.add("active");
                }});
            }});
            document.querySelector(".pill-btn").classList.add("active");
            document.querySelector(".section-panel").classList.add("active");
        }});
    </script>
    <div>
        <h1 class="banner {outcome.split(' ')[0].lower()}">{outcome}</h1>
        <div><img src="https://github.com/mever-team/FairBench/blob/main/docs/fairbench.png?raw=true" alt="logo" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 48px;"/> <h1>based on specific concerns by FairBench</h1></div>
        <div class="pill-buttons">
            <div class="pill-btn" data-target="whatis">What is this?
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/question.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="warning">Responsible use
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/warning.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="process">Analysis methodology
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/methodology.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="pipeline">Data pipeline
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/data.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="details">For experts
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/chart.png?raw=true" height="128px"/>
            </div>
        </div>
        <hr>
        <div id="whatis" class="section-panel">
            We analysed how {getattr(fb.measures, fb_measures[base_measure]).descriptor.details.lower()} is 
            distributed in a model's outputs given a tested dataset by comparing several protected groups 
            {compare_groups.lower()}. 
            {'Expert interpretation of numeric details is required.' if problematic_deviation == 0 else 
            'The assessment depends on specific parameters provided as inputs.'}
        </div>
        <div id="warning" class="section-panel">
            {on_results}
        </div>
        <div id="pipeline" class="section-panel">
            {dataset_description}
            <br>
            <br>
            {model_description}
        </div>
        <div id="process" class="section-panel">
            <p>The {reduction.lower()} of {getattr(fb.measures, fb_measures[base_measure]).descriptor.details.lower()} 
            is obtained across all protected groups, by comparing them {compare_groups.lower()}.
            The result is considered biased if it lays <b>{problematic_deviation:.3f}</b> away from its ideal target 
            that would indicate fairness. For example, the ideal target is 0 for differences between measure values, 
            and 1 for values that should be large (e.g., the minimum accuracy across all groups).
            Some metrics have no known ideal values.</p>
            <p>The analysis considered <b>{len(sensitive.branches())}</b> protected groups:
            <br><i>{'<br>'.join(sensitive.branches().keys())}</i></p>
        </div>
        <div id="details" class="section-panel">{full_report.replace(metric_name, metric_name.replace("_"," "))}</div>
    </div>
    """

    return HTML(html_content)
