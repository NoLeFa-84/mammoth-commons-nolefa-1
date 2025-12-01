from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML, simplified_formatter
from typing import List, Literal
from mammoth_commons.integration import metric
from mammoth_commons.externals import fb_categories, align_predictions


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def model_card(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Literal["Base", "All", "Subgroups"] = "Base",
    compare_groups: Literal["Pairwise", "To the total population"] = "Pairwise",
    problematic_deviation: float = 0.05,
    show_non_problematic: bool = False,
    min_group_size: int = 1,
    presentation: Literal["Numbers", "Bars"] = "Numbers",
) -> HTML:
    """
    <img src="https://github.com/mever-team/FairBench/blob/main/docs/fairbench.png?raw=true" alt="Based on FairBench"
    style="float: left; margin-right: 5px; margin-bottom: 5px; height: 36px;"/>

    <h3>cover a broad picture of imbalances</h3>

    <p>Generates a fairness and bias report using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. This explores many kinds of bias to paint a broad picture and help you decide on what is problematic
    and what is acceptable behavior. Imbalanced distributions of benefits are uncovered to serve as points of
    discussion of real-world impact. </p>

    <details><summary><i>Details for experts.</i></summary>
    <p>The generated report can be viewed in three different formats, where the model card contains a subset of
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
    ignored in the analysis.</p>
    </details>

    Args:
        intersections: Whether to consider only the provided groups, all non-empty group intersections, or all non-empty intersections while ignoring larger groups during analysis. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected. As an example of intersectional bias <b>[1]</b> race and gender together affect algorithmic performance of commercial facial-analysis systems; worst performance for darker-skinned women demonstrates a compounded disparity that would be missed if the analysis looked only at race or only at gender. <br><br><b>[1]</b> <i>Buolamwini, J., & Gebru, T. (2018, January). Gender shades Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency (pp. 77-91). PMLR.</p>
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
        show_non_problematic: Determine whether deviations less than the problematic one should be shown or not. If they are shown, the coloring scheme is adjusted to identify problematic values as red.
        min_group_size: The minimum number of samples per group that should be considered during analysis - groups with less memers are ignored.
        presentation: Whether to focus on showing numbers or showing accompanying bars for easier comparison. Prefer a number comparison to avoid being influenced by comparisons between incomparable measure values.
    """
    # fb = importlib.import_module("fairbench")
    import fairbench as fb

    reps = fb.reports
    prob = float(problematic_deviation)
    min_group_size = int(min_group_size)
    if isinstance(sensitive, str):
        sensitive = [sens.strip() for sens in sensitive.split(",")]
    assert len(sensitive) != 0, "At least one sensitive attribute should be provided"
    assert 0 <= prob <= 1, "Problematic deviation should be in [0,1]"
    presentation = fb.export.HtmlBars if presentation == "Bars" else fb.export.HtmlTable
    report_type = reps.pairwise if compare_groups == "Pairwise" else reps.vsall
    reject = not bool(show_non_problematic)
    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    sensitive = fb.Dimensions(
        {s: fb_categories(dataset.df[s]) for s in sensitive}, _separator=" "
    )
    if intersections != "Base":
        sensitive = sensitive.intersectional(min_size=min_group_size, delimiter=" - ")
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    assert len(sensitive.branches()) != 0, "Could not find any intersections"

    predictions, labels = align_predictions(predictions, dataset.labels)
    predictions = predictions.columns
    labels = labels.columns if labels else None
    report = report_type(predictions=predictions, labels=labels, sensitive=sensitive)
    problematic = set()
    for col in report.filter(
        fb.investigate.DeviationsOver(prob, prune=True)
    ).depends.values():
        for col2 in col.depends.values():
            for value in col2.depends.values():
                problematic.add(
                    f"{value.descriptor.prototype.details} ({value.descriptor.name})"
                )
    if prob != 0:
        report = report.filter(fb.investigate.DeviationsOver(prob, prune=reject))
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
    return HTML(
        simplified_formatter(
            outcome="biased" if problematic else "fair",
            title=(
                f"Biases in {len(problematic)} types of benefits"
                if problematic
                else "no concerns"
            ),
            technology='<div><img src="https://github.com/mever-team/FairBench/blob/main/docs/fairbench.png?raw=true" alt="logo" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 48px;"/> <h1>based on FairBench reporting</h1></div>',
            about=f"""
                <p>{('Some system performance metrics, which indicate obtained benefits like correct or favorable '
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
                Values deviating more than <b>{prob:.3f}</b> from their ideal target
                were counted as problematic. These deviations guide where deeper inspection is needed.
                The result is considered biased if it lays <b>{prob:.3f}</b> away from its ideal target
                that would indicate fairness. For example, the ideal target is 0 for differences between measure values,
                and 1 for values that should be large (e.g., the minimum accuracy across all groups).
                Some metrics have no known ideal values.</p>
                <p>The analysis considered <b>{len(sensitive.branches())}</b> protected groups:
                <br><i>{'<br>'.join(sensitive.branches().keys())}</i></p></p>
            """,
            pipeline=f"{dataset.to_description()}<br><br>{model.to_description()}",
            experts=f"""
                <details><summary>Summary of measures. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'measure' in key.role) + '</table>'}</i><br></details>
                <details><summary>Summary of reductions. </summary><i>{'<table><tr><th>Name</th><th>Description</th></tr>' + ''.join(f'<tr><td>{key.name}</td><td>{key.details}</td></tr>' for key in report.keys() if 'reduction' in key.role) + '</table>'}</i><br></details>
                <div id="expert-tab-header">{"".join(f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views)}</div>
                <div id="expert-tab-body">{"".join(f'<div id="{key}" class="tabcontent">{value}</div>' for key, value in views.items())}</div>
            """,
        )
    )
