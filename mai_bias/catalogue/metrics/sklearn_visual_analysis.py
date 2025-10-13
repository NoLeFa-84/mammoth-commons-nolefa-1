from mammoth_commons.datasets import CSV
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML
from typing import Dict, List
from mammoth_commons.integration import metric, Options
import numpy as np
from mammoth_commons.integration_callback import notify_progress, notify_end
from mammoth_commons.externals import fb_categories


@metric(
    namespace="mammotheu",
    version="v0049",
    python="3.13",
    packages=(
        "fairbench",
        "scikit-learn",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def sklearn_visual_analysis(
    dataset: CSV,
    model: EmptyModel,
    sensitive: List[str],
    predictor: Options("Logistic regression", "Gaussian naive Bayes") = None,
    intersections: Options("Base", "All", "Subgroups") = "Base",
    compare_groups: Options("Pairwise", "To the total population") = None,
    problematic_deviation: float = 0.1,
    show_non_problematic: bool = True,
    top_recommendations: int = 3,
    min_group_size: int = 1,
) -> HTML:
    """
    <img src="https://fairbench.readthedocs.io/fairbench.png" alt="Based on FairBench" style="float: left; margin-right: 5px; margin-bottom: 5px; width: 80px;"/>

    <p>One way to evaluate the fairness of a dataset is by testing for biases using simple models with limited
    degrees of freedom. This module audits datasets by training such models provided by the
    <a href="https://scikit-learn.org/stable/index.html">scikit-learn</a> library on half of the dataset.
    The second half is then used as test data to assess predictive performance and detect classification
    or scoring biases.</p>

    <p>Test data are used to generate a fairness and bias report with the
    <a href="https://fairbench.readthedocs.io/">FairBench</a> library. If strong biases appear in the simple models
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

    Args:
        predictor: Which simple model should be used.
        intersections: Whether to consider only the provided groups, all non-empty group intersections, or all non-empty intersections while ignoring larger groups during analysis. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected.
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
        show_non_problematic: Determine whether deviations less than the problematic one should be shown or not. If they are shown, the coloring scheme is adjusted to identify non-problematic values as green and the rest as either orange or red.
        top_recommendations: The number of top recommendations in evaluation that emulates showing the respective data samples to users when querying the trained model to give examples for each class in the dataset. Common values in the literature are 1,3,5,10.
        min_group_size: The minimum number of samples per group that should be considered during analysis - groups with less memers are ignored.
    """
    import fairbench as fb
    from sklearn import model_selection
    import webbrowser
    import tempfile
    import tempfile, webbrowser

    class HTMLHeatMap:
        def __init__(
            self, legend=True, open_in_browser=True, bar_mode=True, cell_width_px=80
        ):
            self.legend = legend
            self.open_in_browser = open_in_browser
            self.accumulated_bars = []
            self.row_names = {}
            self.col_names = {}
            self.last_title = []
            self.last_supertitle = None
            self.bars = []
            self.level = 0
            self.results = dict()
            self.bar_mode = bar_mode
            self.cell_width_px = cell_width_px
            self.figures_data = []  # (z, text_data, x_labels, y_labels, title)

        # ------------------------------------------------------------------ #
        # internal data accumulation
        # ------------------------------------------------------------------ #
        def _ensure_cell_structure(self):
            for r_idx in range(len(self.accumulated_bars), len(self.row_names)):
                self.accumulated_bars.append(["---"] * len(self.col_names))
            for r_idx, row in enumerate(self.accumulated_bars):
                if len(row) < len(self.col_names):
                    self.accumulated_bars[r_idx] += ["---"] * (
                        len(self.col_names) - len(row)
                    )

        def _embed_bars(self):
            last_title = self.last_title[-1]
            for title, units, val, target in self.bars:
                title = " ".join(word for word in title.split() if word != last_title)

                if title not in self.row_names:
                    row_num = len(self.row_names)
                    self.row_names[title] = row_num
                    self._ensure_cell_structure()
                    self.accumulated_bars.append(["---" for _ in self.col_names])
                else:
                    row_num = self.row_names[title]

                if last_title not in self.col_names:
                    col_num = len(self.col_names)
                    self.col_names[last_title] = col_num
                    for row in self.accumulated_bars:
                        row.append("---")
                else:
                    col_num = self.col_names[last_title]

                self._ensure_cell_structure()
                assert (
                    self.accumulated_bars[row_num][col_num] == "---"
                ), f"Two or more conflicting values for '{title}' under header '{last_title}'."
                self.accumulated_bars[row_num][col_num] = (float(val), float(target))

            self.bars = []
            self.last_supertitle = self.last_title

        def _embed_accumulated_bars(self):
            if not self.accumulated_bars:
                return

            x_labels = list(self.col_names.keys())
            y_labels = list(self.row_names.keys())
            z, text_data = [], []

            for row in self.accumulated_bars:
                z_row, text_row = [], []
                for cell in row:
                    if cell == "---":
                        z_row.append(None)
                        text_row.append("")
                    else:
                        val, target = cell
                        z_row.append((val, target))  # store both
                        text_row.append(str(round(val, 3)))
                z.append(z_row)
                text_data.append(text_row)

            title = " ".join(self.last_supertitle[:-1]) if self.last_supertitle else ""
            self.figures_data.append((z, text_data, x_labels, y_labels, title))

            self.accumulated_bars = []
            self.row_names = {}
            self.col_names = {}

        # ------------------------------------------------------------------ #
        # color computation (green-red gradient based on deviation sign)
        # ------------------------------------------------------------------ #
        @staticmethod
        def _color_for(diff, min_val, max_val):
            # normalize to 0-1 range based on magnitude (global scale)
            ratio = abs(diff - min_val) / (max_val - min_val + 1e-9)
            # clamp
            ratio = max(0.0, min(1.0, ratio))

            if diff < 0:
                # greenish for below target
                r = int(128 * (1 - ratio) + 122)
                g = int(255 - 80 * ratio)
                b = int(164)
            elif diff > 0:
                # reddish for above target
                r = int(255)
                g = int(164 * (1 - ratio) + 60)
                b = int(164 * (1 - ratio) + 60)
            else:
                # neutral
                r, g, b = 200, 200, 200

            return f"rgb({r},{g},{b})"

        # ------------------------------------------------------------------ #
        # main builder
        # ------------------------------------------------------------------ #
        def title(self, text, level=0, link=None):
            if self.bars:
                try:
                    self._embed_bars()
                except AssertionError:
                    self._embed_accumulated_bars()
                    self._embed_bars()
            self.last_title = self.last_title[:level]
            self.last_title.append(text)
            self.level = level
            return self

        def bar(self, title, val: float, target: float, units: str = ""):
            if units == title:
                units = ""
            self.bars.append((title, units, val, target))
            return self

        def end(self):
            self._embed_accumulated_bars()
            return self

        # ------------------------------------------------------------------ #
        # rendering
        # ------------------------------------------------------------------ #
        def show(self, ratio: float = None):
            if not self.figures_data:
                return ""

            # ----------------------------------------------------------
            # Compute global ranges
            # ----------------------------------------------------------
            # Used for coloring (based on deviation)
            all_diffs = [
                abs(val - target)
                for z, *_ in self.figures_data
                for row in z
                for cell in row
                if cell is not None
                for val, target in [cell]
            ]
            min_diff, max_diff = (
                (min(all_diffs), max(all_diffs)) if all_diffs else (0, 1)
            )

            # Used for width normalization (based on raw values)
            all_vals = [
                val
                for z, *_ in self.figures_data
                for row in z
                for cell in row
                if cell is not None
                for val, _ in [cell]
            ]
            max_val = max(all_vals) if all_vals else 1.0

            # User-provided scaling
            if ratio is not None and ratio < 1:
                user_ratio = ratio
            else:
                user_ratio = 1.0

            # ----------------------------------------------------------
            # Render all figures
            # ----------------------------------------------------------
            html_sections = []
            for z, text_data, x_labels, y_labels, title in self.figures_data:
                rows_html = []
                for y, (z_row, text_row) in enumerate(zip(z, text_data)):
                    row_label = y_labels[y] if y < len(y_labels) else f""
                    cells = []
                    for cell, text in zip(z_row, text_row):
                        if cell is None:
                            cell_html = f'<td style="text-align:left;padding:6px;width:{self.cell_width_px}px;"></td>'
                        else:
                            val, target = cell
                            diff = val - target
                            color = self._color_for(diff, min_diff, max_diff)

                            # ----------------------------------------------------------
                            # Width logic:
                            # values ≤ 1: use directly
                            # values > 1: normalize by global max
                            # then apply optional user scaling (<1)
                            # ----------------------------------------------------------
                            if val <= 1:
                                ratio_val = val * user_ratio
                            else:
                                ratio_val = (val / max_val) * user_ratio

                            ratio_val = max(0.0, min(1.0, ratio_val))  # clamp to [0,1]

                            bar_html = f"""
                            <div style="position:relative;width:{self.cell_width_px}px;height:14px;">
                                <div style="position:absolute;bottom:0;left:0;height:100%;width:100%;
                                            background:#ccc;border-radius:2px;overflow:hidden;border:1px solid #000;">
                                    <div style="position:absolute;top:0;left:0;height:100%;
                                                width:{ratio_val * 100:.1f}%;
                                                background:{color};border-radius:2px 0 0 2px;"></div>
                                    <div style="position:relative;z-index:1;width:100%;height:100%;
                                                display:flex;align-items:center;justify-content:center;
                                                font-size:11px;font-family:sans-serif;">
                                        {text}
                                    </div>
                                </div>
                            </div>
                            """
                            cell_html = f'<td style="border:0;padding:2px;text-align:center;width:{self.cell_width_px}px;">{bar_html}</td>'
                        cells.append(cell_html)
                    rows_html.append(
                        f"<tr><th style='text-align:left;'>{row_label}</th>{''.join(cells)}</tr>"
                    )

                header_cells = "".join(
                    f"<th style='width:{self.cell_width_px}px;'>{x}</th>"
                    for x in x_labels
                )
                html_sections.append(
                    f"""
                <div style="margin:20px;">
                    <h2 style="font-family:sans-serif;">{title}</h2>
                    <table style="border-collapse:collapse;font-family:sans-serif;">
                        <tr><th></th>{header_cells}</tr>
                        {''.join(rows_html)}
                    </table>
                </div>
                """
                )

            html_page = f"<html><body>{''.join(html_sections)}</body></html>"
            return html_page

        # ------------------------------------------------------------------ #
        # compatibility stubs
        # ------------------------------------------------------------------ #
        def navigation(self, text, routes: dict):
            return self

        def first(self):
            return self

        def quote(self, text, keywords=()):
            return self

        def result(self, title, val, target, units=""):
            self.results[title] = f"{val:0.3f} {units}"
            return self

        def bold(self, text):
            return self

        def text(self, text):
            return self

        def p(self):
            return self

        def curve(self, *args, **kwargs):
            pass

    min_group_size = int(min_group_size)
    assert len(sensitive) != 0, "Set at least one sensitive attribute"
    reject = not bool(show_non_problematic)
    X = dataset.to_pred(sensitive)
    y = dataset.labels
    y = y[list(y.__iter__())[0]]

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

    views = {
        "Summary": report.show(
            env=HTMLHeatMap(open_in_browser=False),
            depth=1,
        ),
        "Details": report.min.rebase(fb.core.Descriptor("per group", "per group")).show(
            env=HTMLHeatMap(open_in_browser=False),
            depth=1,
        ),
        "Base quantities": report.min.rebase(
            fb.core.Descriptor("per group", "per group")
        ).show(
            env=HTMLHeatMap(open_in_browser=False),
            depth=2,
        ),
    }

    # Generate tabbed HTML content
    tab_headers = "".join(
        f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views
    )
    tab_contents = "".join(
        f'<div id="{key}" class="tabcontent">{value}</div>'
        for key, value in views.items()
    )

    dataset_desc = ""
    if dataset.description is not None:
        dataset_desc += "<h1>Dataset</h1>"
        if isinstance(dataset.description, str):
            dataset_desc += dataset.description + "<br>"
        elif isinstance(dataset.description, dict):
            for key, value in dataset.description.items():
                dataset_desc += f"<h3>{key}</h3>" + value.replace("\n", "<br>") + "<br>"
        else:
            raise Exception(
                f"Dataset description must be a string or a dictionary, not {type(dataset.description)}."
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
       <h1>Audit of {len(sensitive.branches())} groups</h1>
       <hr/>
       <div class="faq-container">
           <div class="faq-box">
                  <h3>❓ What is this?</h3>
                  <p>This is an audit of your dataset using deliberately weak models from scikit-learn
                  (either Logistic Regression or Gaussian Naive Bayes). These simple models are chosen
                  to surface strong correlations that may reveal biases. If such models exhibit bias,
                  more complex models (e.g., deep learning) will likely also carry or amplify them.</p>
                  <br/>
                  <p>The goal is to provide a fairness perspective across sensitive attributes,
                  using the FairBench library for reporting.</p>
            </div>
            <div class="faq-box">
                  <h3>❗ Summary</h3>
                   <p>A fairness report was generated by training and testing a {predictor} classifier on the dataset.
                   Results were computed across {len(sensitive.branches())} protected groups, considering both classification
                   and top-{top_recommendations} recommendation performance.
                   {'Set a problematic deviation parameter for this analysis to simplify what is shown or control coloring thresholds.' if problematic_deviation == 0 else f'Only those that differ at least {problematic_deviation:.3f} from their ideal values are {"shown" if reject else "highlighted in orange or red"}; this is the problematic deviation threshold of the analysis.'}
                   Ideal targets are 0 for metrics that need to be minimized and 1 for those that need to be maximized.
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
