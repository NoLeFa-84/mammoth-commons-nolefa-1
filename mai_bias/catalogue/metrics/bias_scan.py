import mammoth_commons.integration
from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric


@metric(
    namespace="mammotheu",
    version="v053",
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
    scoring: mammoth_commons.integration.Options(
        "Bernoulli", "Gaussian", "Poisson", "BerkJones"
    ) = "Bernoulli",
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
        sensitive = [sens.strip() for sens in sensitive.split(",")]

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
            else:
                text += f'<h4 class="text-warning">Rerunning for new sensitive attributes</h4>'

    faq_style = """
        <div class="container">
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
    {'<h1 class="text-success">No concern</h1>' if counts==0 else '<h1 class="text-danger">Biased intersections of up to '+str(counts)+' attributes</h1>'}
    {faq_style}
    <hr/>
    <div class="faq-container">
        <div class="faq-box">
              <h3>❓ What is this?</h3>
              <p>This is a suggestion of potentially biased attribute intersections, computed with a MAI-BIAS module 
              using the AIF360 library. Results correspond to specific dataset and model loaders and parameters.</p>
              <br/>
              <p>Attributes or combinations of attributes that contain potentially sensitive groups may be used as 
              sensitive attributes by other modules to examine other quantitative aspects. There is a different analysis 
              for each prediction class.</p>
        </div>
        <div class="faq-box">
              <h3>❗ Summary</h3>
                {"" if len(dataset.num) == 0 else "<p><b>Numeric attributes have been ignored; the scan can work with only categorical ones.</b></p>"}
                <p>After scanning for imbalances, the following attribute combinations out of those that were
                <i>not</i> already marked as sensitive were found to be underestimated. 
                {'The scan was run in discovery mode, so the process added all indicated sensitive attributes to sensitive ones and retrying the analysis. This was repeated until no more suspicions were shed on data.' if discovery else 'There may be more attribute combinations that could be underestimated, but only the top one is presented here.'}
                Not all found attributes should necessarily be protected, and you can simplify the problem setting
                by accounting only for the discovered intersection by adding data annotations.</p>
                <br>
                <p><b>{'No biased intersections"' if counts==0 else 'Biased intersections of up to '+str(counts)+' attributes'} were found.</b></p>
        </div>
    </div>
    <hr>
    {text}
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">",
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>"
    </div>
    """
    return HTML(html)
