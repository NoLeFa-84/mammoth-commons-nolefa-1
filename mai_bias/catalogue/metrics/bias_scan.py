import mammoth_commons.integration
from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import List, Literal
from mammoth_commons.integration import metric


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=(
        "aif360",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def bias_scan(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    penalty: float = 0.5,
    scoring: Literal["Bernoulli", "Gaussian", "Poisson", "BerkJones"] = "Bernoulli",
    discovery: bool = True,
) -> HTML:
    """<p>This module scans your dataset to estimate the most biased attributes or combinations of attributes.
    You can use those as inputs to other modules.
    For example, gender may only show bias when combined with socioeconomic status, despite the latter not
    bein inherently sensitive. If you have already marked some
    attributes as sensitive (such as race or gender), the module will **exclude** them from the scan. This allows
    searching for additional patterns that contribute to unfair outcomes.</p>

    <p>A paper describing how this approach estimates biased intersection candidates in **linear** rather
    than **exponential** time is available <a href="https://arxiv.org/pdf/1611.08292">here</a>. Instead of checking
    every possible combination (which can be very time-consuming), it uses a more efficient method.</p>

    <p>To get started, run the module without setting any sensitive attributes. After the first scan,
    advanced users can review the results and mark any problematic attributes it identifies as sensitive.
    Then, run the scan again to uncover additional potential issues—these may be less prominent but still
    worth investigating.</p>

    <p>For convenience, there is a <i>discovery</i> mode available in the parameters. This automatically adds
    attributes suspected of contributing to bias to the list of ignored (already known sensitive) ones, then reruns
    the scan. While this automation helps streamline the process, it removes all attributes contributing to biased
    intersections. A domain expert may prefer to manually remove one attribute at a time by adding it to known
    sensitive attributes and rerun the module to investigate more granular effects on the results.</p>

    Args:
        penalty: A positive. The higher the penalty, the less complex the highest scoring subset that gets returned is, but penalties as small as 1.E-12 could also be acceptable to promote finding intersections of many attributes.
        scoring: The distribution used to compute p-values. Can be Bernoulli, Gaussian, Poisson, or BerkJones.
        discovery: Whether the scan should attempt to create a list of problematic attribute combinations in decreasing order of importance. That list will contain only non-overlapping attribute intersections.
    """
    import pandas as pd
    from aif360.sklearn.detectors import bias_scan as aif360bias_scan

    if isinstance(sensitive, str):
        sensitive = [sens.strip() for sens in sensitive.split(",") if sens.strip()]

    predictions = pd.Series(model.predict(dataset, sensitive))
    dataset = dataset.to_csv(sensitive)
    penalty = float(penalty)
    text = ""

    counts = 0
    starting_sensitive = sensitive
    for label in dataset.labels:
        sensitive = starting_sensitive
        text += f'<h2 class="text-secondary">Prediction label: {label}</h2>'
        while True:
            labels = pd.Series(dataset.labels[label])
            cats = [cat for cat in dataset.cat if cat not in sensitive]
            if len(cats) == 0 and text:
                text += f"<i>All categorical attributes are already considered sensitive</i>"
                break
            assert len(cats), "All categorical attributes already known as sensitive"
            text += (
                f"<i>Already known sensitive attributes to be ignored: {', '.join(sensitive)}</i>"
                if sensitive
                else f"<i>No attributes to be ignored (scanning everything)</i>"
            )
            X = dataset.df[cats]
            ret = aif360bias_scan(
                X=X,
                y_true=labels,
                y_pred=predictions,
                overpredicted=False,
                scoring=scoring,
                penalty=penalty,
            )
            ret = ret[0]
            stext = ""
            for key, values in ret.items():
                for value in values:
                    stext += f"<tr><td>{key}</td><td>{value}</td></tr>"
                sensitive = sensitive + [key]
            if len(ret) == 0:
                text += "<p>No suspicious attribute intersection</p>"
                break
            text += '<div class="table-responsive"><table class="table table-striped table-bordered table-hover mt-3">'
            text += '<thead class="thead-dark"><tr><th>Attribute</th><th>Value</th></tr></thead><tbody>'
            text += stext
            text += "</tbody></table></div>"
            counts = max(counts, len(ret))
            if not discovery:
                break
            text += (
                f'<h4 class="text-warning">Rerunning for new sensitive attributes</h4>'
            )

    outcome = "No concerns" if counts == 0 else f"{counts} attribute biases"
    outcome_class = "fair" if counts == 0 else "biased"
    dataset_description = dataset.to_description().split("Args:")[0]
    model_description = model.to_description().split("Args:")[0]

    html_content = f"""
    <style>
        .pill-buttons {{display: flex; gap: 12px; margin: 20px 0;}}
        .banner {{width: 100%;  padding: 18px 24px; font-size: 42px; font-weight: 700; text-align: center; color: white; border-radius: 12px; margin-bottom: 25px;}}
        .banner.fair {{ background: #2e8b57; }}
        .banner.biased {{ background: #c0392b; }}
        .pill-btn {{ width:100%; text-align:center; padding: 10px 18px; background: #f5f5f5; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; font-size: 18px; transition: background 0.2s;}}
        .pill-btn:hover {{ background: #e0e0e0; }}
        .pill-btn.active {{ background: #d0d0d0; border-color: #999;}}
        .section-panel {{ display: none; border: 0px solid #ddd; padding: 0px; border-radius: 8px; background: white; }}
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
        <h1 class="banner {outcome_class}">{outcome}</h1>

        <div class="pill-buttons">
            <div class="pill-btn" data-target="whatis">What is this?
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/question.png?raw=true" height="128px"/>
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

        <div id="whatis" class="section-panel">
            <p>This module identifies potentially biased intersections of attributes using 
            IBM's AIF360 bias scan detector. Already-known sensitive attributes are ignored during scanning. Remaining 
            attributes (including non-sensitive ones) are tested for imbalances that may 
            contribute to unfair predictions. There is a separate analysis for each prediction class.</p>
            {'The following' if sensitive else 'No'} attributes exhibited biases in some of their values or during the 
            intersection with other attributes{':' if sensitive else '.'}  
            <br><i>{'<br>'.join(sensitive)}</i>
        </div>

        <div id="process" class="section-panel">
            <p>The scan evaluates attribute combinations by computing p-values under a
            <b>{scoring}</b> statistical model. A penalty parameter <b>{penalty}</b> controls the complexity of
            discovered intersections: higher penalty → simpler intersections.
            {'In discovery mode, detected suspicious attributes are added to the ignored list and the scan repeated until no more intersections are detected.'
            if discovery else
            'Only the top suspicious attribute combination is reported; further combinations may exist.'}
            </p>
        </div>
        <div id="pipeline" class="section-panel">{dataset_description} <br><br> {model_description}</div>
        <div id="details" class="section-panel">{text}</div>
    </div>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    return HTML(html_content)
