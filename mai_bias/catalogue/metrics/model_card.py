import importlib

from mammoth_commons.datasets import Dataset, Labels
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import Dict, List
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v0049",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def model_card(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Options("Base", "All", "Subgroups") = "Base",
    compare_groups: Options("Pairwise", "To the total population") = None,
    problematic_deviation: float = 0.1,
    show_non_problematic: bool = True,
    min_group_size: int = 1,
    presentation: Options("Numbers", "Bars") = "Numbers",
) -> HTML:
    """
    <img src="https://fairbench.readthedocs.io/fairbench.png" alt="Based on FairBench"
    style="float: left; margin-right: 5px; margin-bottom: 5px; width: 80px;"/>

    <p>Generates a fairness and bias report using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. This explores many kinds of bias to paint a broad picture and help you decide on what is problematic
    and what is acceptable behavior.
    The generated report can be viewed in three different formats, where the model card contains a subset of
    results but attaches to these socio-technical concerns to be taken into account:</p>
    <ol>
        <li>A summary table of results.</li>
        <li>A simplified model card that includes concerns.</li>
        <li>The full report, including details.</li>
    </ol>

    <p>The module's report summarizes how a model behaves on a provided dataset across different population groups.
    These groups are based on sensitive attributes like gender, age, and race. Each attribute can have multiple values,
    such as several genders or races. Numeric attributes, like age, are normalized to the range [0,1] and treated
    as fuzzy values, where 0 indicates membership to a fuzzy group of "small" values, and 1 indicates membership to
    a fuzzy group of "large" values. A separate set of fairness metrics is calculated for each prediction label.</p>

    <p>If intersectional subgroup analysis is enabled, separate subgroups are created for each combination of sensitive
    attribute values. However, if there are too many attributes, some groups will be small or empty. Empty groups are
    ignored in the analysis. The report may also include information about built-in datasets.</p>

    Args:
        intersections: Whether to consider only the provided groups, all non-empty group intersections, or all non-empty intersections while ignoring larger groups during analysis. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected.
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
        show_non_problematic: Determine whether deviations less than the problematic one should be shown or not. If they are shown, the coloring scheme is adjusted to identify problematic values as red.
        min_group_size: The minimum number of samples per group that should be considered during analysis - groups with less memers are ignored.
        presentation: "Whether to focus on showing numbers or showing accompanying bars for easier comparison. Prefer a number comparison to avoid being influenced by comparisons between incomparable measure values.
    """
    fb = importlib.import_module("fairbench")
    reps = fb.reports
    prob = float(problematic_deviation)
    min_group_size = int(min_group_size)
    assert len(sensitive) != 0, "At least one sensitive attribute should be provided"
    assert 0 <= prob <= 1, "Problematic deviation should be in [0,1]"
    presentation = fb.export.HtmlBars if presentation == "Bars" else fb.export.HtmlTable
    report_type = reps.pairwise if compare_groups == "Pairwise" else reps.vsall
    reject = not bool(show_non_problematic)
    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    sensitive = fb.Dimensions({s: fb_categories(dataset.df[s]) for s in sensitive})
    if intersections != "Base":
        sensitive = sensitive.intersectional(min_size=min_group_size)
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    assert len(sensitive.branches()) != 0, "Could not find any intersections"

    predictions, labels = align_predictions(predictions, dataset.labels)
    predictions = predictions.columns
    labels = labels.columns if labels else None
    report = report_type(predictions=predictions, labels=labels, sensitive=sensitive)
    if prob != 0:
        report = report.filter(fb.investigate.DeviationsOver(prob, prune=reject))

    views = {
        "Summary": report.show(env=presentation(view=False, filename=None)),
        "Stamps": report.filter(fb.investigate.Stamps).show(
            env=fb.export.Html(view=False, filename=None),
            depth=2 if isinstance(predictions, dict) else 1,
        ),
        "Full report": report.show(
            env=presentation(view=False, filename=None),
            depth=3 if isinstance(predictions, dict) else 2,
        ),
    }
    tab_headers = "".join(
        f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views
    )
    tab_contents = "".join(
        f'<div id="{key}" class="tabcontent">{value}</div>'
        for key, value in views.items()
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

    html_content = f"""
       {faq_style}
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
       <div class="container">
       <h1>Report for {len(sensitive.branches())} groups</h1>
       <hr/>
       <div class="faq-container">
           <div class="faq-box">
                  <h3>❓ What is this?</h3>
                  <p>This is a broad view of group imbalances, computed with a MAI-BIAS module 
                  using the FairBench library. Results are organized into a concise tabular summary,
                  a fairness model card that presents popular parts of this analysis, and into
                  details of the analysis. Analysis always considers an aggregate value of group comparisons.</p>
                  <br/>
                  <p>The goal of this analysis is to help gain a broad picture. Thresholds that affect which values 
                  are shown, colors, and checkmarks/crosses are only there to help identify where analysis should start
                  from. It is normal that some biases not important to the application context are found, in which
                  case you need to assess whether they are actually irrelevant.</p>
            </div>
            <div class="faq-box">
                  <h3>❗ Summary</h3>
                   <p>A report was generated over several prospective biases to paint a broad 
                   picture{'; set a problematic deviation parameter for this analysis to simplify what is shown or control coloring thresholds.' if prob == 0 else f', but for simplicity only those that differ at least {prob:.3f} from their ideal values are {"shown" if reject else "colored orange or red, otherwise green"}; this is the problematic deviation parameter of the analysis.'}
                   Ideal targets are 0 for values that need to be small and 1 for those that need to be large. For some report entries, ideal targets are unknown.
                   </p>
                   <br>
                   <p>
                   Presented values combine a base performance measure, computed on each group or subgroup with at least {min_group_size} members, and an aggregated value across all data samples.
                   Switch to "Details" to see full descriptions of the measures as well as the distributions across groups.
                   Results may not give the full picture, and not all biases may be harmful to the social context. Switch to "Stamps" so see popular
                   literature definitions alongside caveats and recommendations.
                   </p>
                   <br>
                   <details><summary>In total {len(sensitive.branches())} protected groups were analysed. </summary><i>{', '.join(sensitive.branches().keys())}</i><br></details>
                   <details><summary>Summary of measures. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'measure' in key.role) + '</table>'}</i><br></details>
                   <details><summary>Summary of reductions. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'reduction' in key.role) + '</table>'}</i><br></details>
                   <br><p><b>Results require manual inspection to determine which values are socially or contextually problematic.</b></p>
            </div>
       </div> 
       <hr/>
       <div id="tab-header-container">{tab_headers}</div>
       <div id="tab-content-container">{tab_contents}</div>
       <div style="clear: both;">{dataset.to_description()}</div>
       <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const tabContainer = document.getElementById("tab-header-container");
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
        </div>
       """

    return HTML(html_content)
