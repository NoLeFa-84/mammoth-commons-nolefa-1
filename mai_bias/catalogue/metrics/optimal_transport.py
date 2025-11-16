from mammoth_commons.datasets import Dataset, ImageLike
from mammoth_commons.externals import align_predictions
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
        "aif360[OptimalTransport]",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
        "scikit-image",
    ),
)
def optimal_transport(
    dataset: Dataset, model: Predictor, sensitive: List[str], threshold: float = 0.01
) -> HTML:
    """
    Creates an optimal transport evaluation based on the implementation provided by the AIF360 library.
    The evaluation computes the Wasserstein distance that reflects the cost of transforming the predictive
    distributions between sensitive attribute groups.

    <p>Optimal Transport (OT) is a field of mathematics which studies the geometry of probability spaces. Among its
    many contributions, OT provides a principled way to compare and align probability distributions by taking into
    account the underlying geometry of the considered metric space.
    As a mathematical problem, it was first introduced by Gaspard Monge in 1781. It addresses the task of determining
    the most efficient method for transporting mass from one distribution to another. In this problem, the cost
    associated with moving a unit of mass from one position to another is referred to as the ground cost. The primary
    objective of OT is to minimize the total cost incurred when moving one mass distribution onto another.
    </p><p>
    OT can be used to detect model-induced bias by calculating the a cost known as Earth Mover's distance or
    Wasserstein distance between the distribution of ground truth labels and model predictions for each of the
    protected groups. If its value is close to 1, the model is biased towards this group.
    </p>

    <b>License</b><p><i>Parts of the above description are adapted from AIF360
    (<a href="https://github.com/Trusted-AI/AIF360">https://github.com/Trusted-AI/AIF360</a>),
    which is licensed under Apache License 2.0.</i></p>

    Args:
        threshold: Transport distances below the given threshold are considered negligible.
    """
    from aif360.sklearn.metrics import ot_distance
    import pandas as pd

    assert len(sensitive) != 0, "At least one sensitive attribute should be selected"
    threshold = float(threshold)
    text = ""
    predictions = pd.Series(model.predict(dataset, sensitive))
    dataset = dataset.to_csv(sensitive)
    labels = dataset.labels
    predictions, labels = align_predictions(predictions, labels)
    predictions = predictions.columns
    labels = labels.columns if labels else None

    worst_distance = 0
    offenders = list()
    text += """
    <div class="container mt-4">
    <table class="table table-striped table-bordered">
    <thead class="table-dark"><tr><th>Attribute</th><th>Group</th>
    """
    for label_name in labels:
        text += f"<th>{label_name} distance</th>"
    text += "</tr></thead><tbody>"

    # Collect distances for merging
    results = {}
    for label_name in labels:
        label = pd.Series(labels[label_name])
        for attr in sensitive:
            df = dataset.df[attr]
            if hasattr(predictions, "numpy"):
                predictions = predictions.numpy()
            if hasattr(df, "numpy"):
                df = df.numpy()
            dist = ot_distance(
                y_true=label, y_pred=pd.Series(predictions[label_name]), prot_attr=df
            )
            for k, v in dist.items():
                if (attr, k) not in results:
                    results[(attr, k)] = {}
                results[(attr, k)][label_name] = v

    # Render merged table
    for (attr, group), distances in results.items():
        text += f"<tr><td>{attr}</td><td>{group}</td>"
        for label_name in labels:
            if distances.get(label_name, 0) > threshold:
                offenders.append(f"{attr} {group} for target {label_name}")
            worst_distance = max(distances.get(label_name, 0), worst_distance)
            text += f"<td>{distances.get(label_name, 'N/A'):.3f}</td>"
        text += "</tr>"
    text += "</tbody></table></div>"
    offenders = (
        (
            f"<h2 class='text-danger'>Distances over threshold</h2>-"
            + "<br>-".join(set(offenders))
        )
        if offenders
        else "<i>No bias concerns found (this does not mean that there are none)</i>"
    )
    message = f"Bias detected" if worst_distance > threshold else f"No concern"
    text = (
        f"""
    <div class="container mt-4">
        <h1 class={"text-danger" if worst_distance>=threshold else "text-primary"}>{message}</h1>
        <p>
            The normalized Wasserstein distance is computed for each group based on optimal transport theory. 
            Higher values (maximum is 1, minimum is 0) indicate greater 
            distribution differences between each group and the rest of the population. 
            Differences more than the manually provided threshold {threshold:.3f} are considered to indicate bias.
        </p>
    </div>
    """
        + text
        + f""""<div class="container">{offenders}</div>"""
        + """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """
    )

    return HTML(text)
