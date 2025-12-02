import importlib
import numpy as np
import sklearn as sk

from mammoth_commons.datasets import Dataset, Labels
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import Dict, List
from mammoth_commons.integration import metric, Options
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v0048",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def nolefa_ts_binary_classification(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Options("Base", "All", "Subgroups") = "Base",
    compare_groups: Options("Pairwise", "To the total population") = None,
    problematic_deviation: float = 0.1,
    show_non_problematic: bool = True,
    min_group_size: int = 1,
) -> HTML:
    """
    NoLeFa Testing Suite Binary Classification

    """
    # Extract labels from test data set
    labels = np.array(dataset.labels.columns.get('yes'))

    # Run model
    prediction = model.predict(dataset, None)

    # Calculate metrics comparing model results with labels
    precision = sk.metrics.precision_score(labels, prediction)
    recall = sk.metrics.recall_score(labels, prediction)
    accuracy = sk.metrics.accuracy_score(labels, prediction)

    probabilities = model.predict_probabilities(dataset, None)


    import pdb; pdb.set_trace()
    fb = importlib.import_module("fairbench")
    reps = fb.reports
    prob = float(problematic_deviation)
    min_group_size = int(min_group_size)
    assert len(sensitive) != 0, "At least one sensitive attribute should be provided"
    assert 0 <= prob <= 1, "Problematic deviation should be in [0,1]"
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

    import pdb; pdb.set_trace()
    report = report_type(predictions=predictions, labels=labels, sensitive=sensitive)
    if prob != 0:
        report = report.filter(fb.investigate.DeviationsOver(prob, prune=reject))
    
    views = {
        "Summary": report.show(env=fb.export.HtmlTable(view=False, filename=None)),
        "Stamps": report.filter(fb.investigate.Stamps).show(
            env=fb.export.Html(view=False, filename=None),
            depth=2 if isinstance(predictions, dict) else 1,
        ),
        "Full report": report.show(
            env=fb.export.Html(view=False, filename=None),
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
