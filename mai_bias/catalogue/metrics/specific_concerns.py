from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import Dict, List
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def specific_concerns(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Options("Base", "All", "Subgroups") = "Base",
    base_measure: Options(
        "Accuracy",
        "True positive rate",
        "True negative rate",
        "Area under curve",
        "Positive rate",
    ) = None,
    compare_groups: Options("Pairwise", "To the total population") = None,
    reduction: Options(
        "Min",
        "Max",
        "Weighted mean",
        "Max difference",
        "Max relative difference",
        "Max betweeness area",
        "Standard deviation x2",
        "Gini coefficient",
    ) = None,
    problematic_deviation: float = 0.1,
) -> HTML:
    """
    <img src="https://fairbench.readthedocs.io/fairbench.png" alt="Based on FairBench" style="float: left; margin-right: 5px; margin-bottom: 5px; width: 80px;"/>

    <p>Computes a fairness or bias measure that matches a specific type of numerical
    evaluation using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. The measure is built by combining simpler options to form more than 300 valid alternatives.</p>

    <span class="alert alert-warning alert-dismissible fade show" role="alert"
    style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i>
    This computes a specific fairness concerns and does not paint a broad enough picture. Make sure that
    you explore prospective biases with other modules first, like <i>model card</i>.</span>

    <p>The assessment is conducted over sensitive attributes like gender, age, and race. Each attribute can have multiple values,
    such as several genders or races. Numeric attributes, like age, are normalized to the range [0,1] and treated
    as fuzzy values, where 0 indicates membership to a fuzzy group of "small" values, and 1 indicates membership to
    a fuzzy group of "large" values. A separate set of fairness metrics is calculated for each prediction label.</p>

    <p>If intersectional subgroup analysis is enabled, separate subgroups are created for each combination of sensitive
    attribute values. However, if there are too many attributes, some groups will be small or empty. Empty groups are
    ignored in the analysis. The report may also include information about built-in datasets.</p>

    Args:
        intersections: Whether to consider only the provided groups (Base), all non-empty group intersections (All), or all non-empty intersections while ignoring larger groups during analysis (Subgroups). For example, the last option may not contain a `White` dimension if `White Men` is an existing dimension. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected.
        base_measure: A base measure of algorithmic performance to be computed on each group.
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        reduction: The strategy with which to reduce all measure comparisons to one value.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
    """
    import fairbench as fb

    assert len(sensitive) != 0, "At least one sensitive attribute should be selected"

    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    sensitive = fb.Dimensions({s: fb_categories(dataset.df[s]) for s in sensitive})
    if intersections != "Base":
        sensitive = sensitive.intersectional()
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    assert len(sensitive.branches()) != 0, "Could not find any intersections"
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
    if problematic_deviation != 0:
        report = report.filter(
            fb.investigate.DeviationsOver(problematic_deviation, prune=False)
        )

    full_report = report.show(
        env=fb.export.Html(view=False, filename=None),
        depth=1 if isinstance(predictions, dict) else 0,
    )

    dataset_desc = dataset.to_description()
    if problematic_deviation == 0:
        outcome = "Report"
    else:
        outcome = (
            "Fair" if report.flatten(True)[0] < problematic_deviation else "Biased"
        )

    faq_html = f"""
    <style>
    .faq-container {{
      max-width: 600px;
      margin: 20px auto;
      font-family: Arial, sans-serif;
    }}
    .faq-box {{
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
      background: #fff;
    }}
    .faq-box h3 {{
      margin-top: 0;
      font-size: 1.2em;
      color: #333;
    }}
    .faq-box p {{
      margin: 0;
      color: #555;
    }}
    </style>

    <div class="faq-container">
        <div class="faq-box">
            <h3>❓ What is this?</h3>
            <p>This module computes a <i>specific fairness concern</i> using the 
            <a href="https://github.com/mever-team/FairBench" target="_blank">FairBench</a> library. 
            </p>
            <br/>
            <p>Unlike broad analyses (e.g. model cards), this focuses narrowly on one fairness 
            definition. It is most useful when you already know what type of disparity you want 
            to evaluate, such as accuracy gaps or differences in positive rates between groups.</p>
        </div>

        <div class="faq-box">
            <h3>❗ Summary</h3>
               <p>A report was generated by MAI-BIAS the bias assessment on metric {metric_name}
               standardized by the <a href="https://github.com/mever-team/FairBench" target="_blank">FairBench</a>
               library. The standardized metric combines a base performance metric, computed on each group or subgroup, 
               and an aggregated value that captures notions of bias or fairness across all of those groups.
               Differences at least {problematic_deviation:.3f} away from their ideal values are colored red, 
               otherwise green. Orange indicates that ideal values are not known a-priori.
               Ideal targets are 0 for values that need to be small, like relative differences, and
                1 for those that need to be large, like minimum accuracy across groups.
               For some metrics, ideal targets are unknown.
               </p>
               <br/>
               <p>
               The analysis combines a base measure (e.g. accuracy, true positive rate), 
               with a method for comparing groups (pairwise or against the population), 
               and how to reduce differences to a single score. Results include per-class
               values and distributions of metric values across groups. Pay attention to the
               caveats and recommendations too.
               </p>
           <br>
           <details><summary>In total {len(sensitive.branches())} protected groups were analysed. </summary>
           <i>{', '.join(sensitive.branches().keys())}</i></details>
           <br>
           <p><b>{'Manual interpretation is required because problematic deviation was zero.' 
            if problematic_deviation==0 else outcome+' model assessment on the provided data.'}</b></p>
        </div>
    </div>
    """

    html_content = f"""
       <style>
           .tablinks {{
               background-color: #ddd;
               padding: 10px;
               cursor: pointer;
               border: none;
               border-radius: 5px;
               margin: 5px;
           }}
           .tablinks:hover {{ background-color: #bbb; }}
           .tablinks.active {{ background-color: #aaa; }}

           .tabcontent {{
               display: none;
               padding: 10px;
               border: 1px solid #ccc;
           }}
           .tabcontent.active {{ display: block; }}
       </style>
       <script>
           document.addEventListener("DOMContentLoaded", function() {{
               const tabContainer = document.querySelector("div");
               tabContainer.addEventListener("click", function(event) {{
                   if (event.target.classList.contains("tablinks")) {{
                       let tabName = event.target.getAttribute("data-tab");
                       document.querySelectorAll(".tablinks").forEach(tab => tab.classList.remove("active"));
                       document.querySelectorAll(".tabcontent").forEach(content => content.classList.remove("active"));
                       event.target.classList.add("active");
                       document.getElementById(tabName).classList.add("active");
                   }}
               }});

               // Show the first tab by default
               let firstTab = document.querySelector(".tablinks");
               if (firstTab) {{
                   firstTab.classList.add("active");
                   document.getElementById(firstTab.getAttribute("data-tab")).classList.add("active");
               }}
           }});
       </script>
       <h1>{outcome} {metric_name}</h1>
       <hr/>
       {faq_html}
       <hr/>
       <br>
       <div>{full_report}</div>
       <div style="clear: both;">{dataset_desc}</div>
       """.replace(
        metric_name, "<i>" + metric_name.replace("_", " ") + "</i>"
    )

    return HTML(html_content)
