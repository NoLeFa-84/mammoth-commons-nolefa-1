from mammoth_commons.datasets import CSV
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML, simplified_formatter
from typing import List, Literal
from mammoth_commons.integration import metric
import numpy as np
from mammoth_commons.integration_callback import notify_progress, notify_end
from mammoth_commons.externals import fb_categories
from mammoth_commons.reminders import on_results, logo_fairbench


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=(
        "fairbench",
        "scikit-learn",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
    logo=logo_fairbench,
)
def sklearn_audit(
    dataset: CSV,
    model: EmptyModel,
    sensitive: List[str],
    predictor: Literal[
        "Logistic regression", "Gaussian naive Bayes"
    ] = "Gaussian naive Bayes",
    intersections: Literal["Base", "All", "Subgroups"] = "Subgroups",
    compare_groups: Literal["Pairwise", "To the total population"] = "Pairwise",
    problematic_deviation: float = 0.05,
    show_non_problematic: bool = False,
    top_recommendations: int = 3,
    min_group_size: int = 1,
    presentation: Literal["Numbers", "Bars"] = "Numbers",
) -> HTML:
    """
    <h3>report on the biases of a simple predictor</h3>
    <p>One way to evaluate the fairness of a dataset is by testing for biases using simple models with limited
    degrees of freedom. This module audits datasets by training such models provided by the
    <a href="https://scikit-learn.org/stable/index.html">scikit-learn</a> library on half of the dataset.
    The second half is then used as test data to assess predictive performance and detect classification
    or scoring biases. Test data are used to generate a fairness and bias report with the
    <a href="https://fairbench.readthedocs.io/">FairBench</a> library.
    </p>
    <details><summary><i>Details for experts.</i></summary>
    <p>If strong biases appear in the simple models
    that are explored, they may also persist in more complex models trained on the same data. To focus on the most
    significant biases, adjust the minimum shown deviation parameter.
    The report provides multiple types of fairness and bias assessments and can be viewed in three different formats,
    where the model card contains a subset of results but attaches to these socio-technical concerns to be taken into
    account:</p>
    <ol>
        <li>A summary table of results.</li>
        <li>A simplified model card with key fairness concerns.</li>
        <li>A full detailed report.</li>
    </ol>

    <p>The module's report summarizes how a model behaves on a provided dataset across different population groups.
    These groups are based on sensitive attributes like gender, age, and race. Each attribute can have multiple values,
    such as several genders or races. Numeric attributes, like age, are normalized to the range [0,1] and treated
    as fuzzy values, where 0 indicates membership to a fuzzy group of "small" values, and 1 indicates membership to
    a fuzzy group of "large" values. A separate set of fairness metrics is calculated for each prediction label.</p>

    <p>If intersectional subgroup analysis is enabled, separate subgroups are created for each combination of sensitive
    attribute values. However, if there are too many attributes, some groups will be small or empty. Empty groups are
    ignored in the analysis. The report may also include information about built-in datasets.</p>
    </details>

    Args:
        predictor: Which simple model should be used.
        intersections: Whether to consider only the provided groups, all non-empty group intersections, or all non-empty intersections while ignoring larger groups during analysis. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected. As an example of intersectional bias <b>[1]</b> race and gender together affect algorithmic performance of commercial facial-analysis systems; worst performance for darker-skinned women demonstrates a compounded disparity that would be missed if the analysis looked only at race or only at gender. <br><br><b>[1]</b> <i>Buolamwini, J., & Gebru, T. (2018, January). Gender shades Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency (pp. 77-91). PMLR.</p>
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
        show_non_problematic: Determine whether deviations less than the problematic one should be shown or not. If they are shown, the coloring scheme is adjusted to identify non-problematic values as green and the rest as either orange or red.
        top_recommendations: The number of top recommendations in evaluation that emulates showing the respective data samples to users when querying the trained model to give examples for each class in the dataset. Common values in the literature are 1,3,5,10.
        min_group_size: The minimum number of samples per group that should be considered during analysis - groups with less memers are ignored.
        presentation: Whether to focus on showing numbers or showing accompanying bars for easier comparison. Prefer a number comparison to avoid being influenced by comparisons between incomparable measure values.
    """
    import fairbench as fb
    from sklearn import model_selection

    min_group_size = int(min_group_size)
    if isinstance(sensitive, str):
        sensitive = [sens.strip() for sens in sensitive.split(",")]
    assert len(sensitive) != 0, "Set at least one sensitive attribute"
    presentation = fb.export.HtmlBars if presentation == "Bars" else fb.export.HtmlTable
    reject = not bool(show_non_problematic)
    X = dataset.to_pred(sensitive)
    y = dataset.labels
    y = y[list(y.__iter__())[0]]
    original_model = model

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        idx_test,
    ) = model_selection.train_test_split(
        X, y, np.arange(0, len(y), dtype=np.int64), test_size=0.2
    )
    if predictor == "Logistic regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import warnings

        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(max_iter=1, warm_start=True)
        for i in range(1000):
            model.fit(X, y)
            notify_progress(
                (i + 1) / 1000,
                f"Training a logistic regression classifier for {i+1}/{1000} epochs",
            )
    else:
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
        batch_size = 100
        unique_classes = np.unique(y)
        for i in range(0, len(y), batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            model.partial_fit(X_batch, y_batch, classes=unique_classes)
            notify_progress(
                min(1.0, (i + batch_size) / len(X)),
                f"Fitting a Gaussian naive Bayes classifier on {i}/{len(y)} data points",
            )
    notify_end()
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    sensitive = fb.Dimensions(
        {attr + " ": fb_categories(dataset.df[attr][idx_test]) for attr in sensitive}
    )
    if intersections != "Base":
        sensitive = sensitive.intersectional(min_size=min_group_size)
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    report_type = (
        fb.reports.pairwise if compare_groups == "Pairwise" else fb.reports.vsall
    )

    report = report_type(
        predictions=predictions,
        labels=np.array(y_test),
        scores=scores,
        sensitive=sensitive,
        top=top_recommendations,
    )
    problematic_deviation = float(problematic_deviation)
    assert (
        0 <= problematic_deviation <= 1
    ), "Minimum problematic deviation should be in the range [0,1]"
    if problematic_deviation != 0:
        report = report.filter(
            fb.investigate.DeviationsOver(problematic_deviation, prune=reject)
        )
    problematic = set()
    for col in report.filter(
        fb.investigate.DeviationsOver(problematic_deviation, prune=True)
    ).depends.values():
        for value in col.depends.values():
            problematic.add(
                f"{value.descriptor.prototype.details} ({value.descriptor.name})"
            )

    views = {
        "Summary": report.show(env=presentation(view=False, filename=None)),
        "Stamps": report.filter(fb.investigate.Stamps).show(
            env=fb.export.Html(view=False, filename=None),
            depth=2 if isinstance(predictions, dict) else 1,
        ),
        "Distribution per group": report.show(
            env=presentation(view=False, filename=None),
            depth=3 if isinstance(predictions, dict) else 2,
        ),
    }
    expert_tabs_header = "".join(
        f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views
    )
    expert_tabs_body = "".join(
        f'<div id="{key}" class="tabcontent">{value}</div>'
        for key, value in views.items()
    )
    html_content = simplified_formatter(
        outcome="biased" if problematic else "fair",
        title=(
            f"{len(problematic)} dataset biases"
            if problematic
            else "No dataset concerns"
        ),
        technology=logo_fairbench + "based on FairBench reporting",
        about=f"""
            <p>This is a dataset audit using a {predictor} model trained on-the-fly. The model is deliberately simple,
            so that, if it exhibits bias, more complex models (e.g., deep learning ones) will likely carry or 
            amplify those biases too.
            {('Some system performance metrics, which indicate obtained benefits like correct or favorable '
              'operation, were found unevenly distributed across the population. '
              'These biases occurred in at least one prediction class and at least one way of aggregating the comparison '
              'among multiple groups. Expert assessment is needed to help understand which biases may be considered unfair. '
              'The biased metrics are:')
            if problematic else 'No biases were found.'}
            <br><br>
            <i>{'<br>'.join(problematic)}</i>
        """,
        methodology=f"""
            <p>Groups were compared <b>{compare_groups.lower()}</b>.
            Values deviating more than <b>{problematic_deviation:.3f}</b> from their ideal target
            were counted as problematic. These deviations guide where deeper inspection is needed.
            The result is considered biased if it lays <b>{problematic_deviation:.3f}</b> away from its ideal target 
            that would indicate fairness. For example, the ideal target is 0 for differences between measure values, 
            and 1 for values that should be large (e.g., the minimum accuracy across all groups).
            Some metrics have no known ideal values.</p>
            <p>The analysis considered <b>{len(sensitive.branches())}</b> protected groups:
            <br><i>{'<br>'.join(sensitive.branches().keys())}</i></p></p>
        """,
        pipeline=f"{dataset.to_description()}<br><br>{original_model.to_description()}",
        experts=f"""
            <details><summary>Summary of measures. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'measure' in key.role) + '</table>'}</i><br></details>
            <details><summary>Summary of reductions. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'reduction' in key.role) + '</table>'}</i><br></details>
            <div id="expert-tab-header">{expert_tabs_header}</div>
            <div id="expert-tab-body">{expert_tabs_body}</div>
        """,
    )
    return HTML(html_content)
