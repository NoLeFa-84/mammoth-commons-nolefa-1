from mammoth_commons.datasets import Dataset, ImageLike
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric
from mammoth_commons.integration_callback import notify_progress, notify_end


def generate_nested_pie_chart(df, columns, title=None, color_scheme=None):
    """
    Generate a nested pie chart (sunburst) where the same values in the same ring
    have the same color while ensuring distinct colors between rings.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(columns, list) or len(columns) < 1:
        raise TypeError("columns must be a list with at least 1 column name")
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Determine a numeric column to use as values
    value_col = None
    for col in df.columns:
        if col not in columns and pd.api.types.is_numeric_dtype(df[col]):
            value_col = col
            break

    # List of color maps to use for each column
    color_palettes = [
        px.colors.qualitative.Set1,
        px.colors.qualitative.Set2,
        # px.colors.qualitative.Set3,
        # px.colors.qualitative.Plotly,
        # px.colors.qualitative.D3
    ]  # Add more palettes if needed

    # Create a color mapping for each level
    color_map = {}
    for j, col in enumerate(columns):
        unique_values = df[col].unique()
        # Select a color palette for the current column, cycling through available palettes
        color_palette = color_palettes[j % len(color_palettes)]
        color_map[col] = {
            val: color_palette[i % len(color_palette)]
            for i, val in enumerate(unique_values)
        }

    # Create the sunburst chart
    fig = px.sunburst(
        df,
        path=columns,
        values=value_col,
        hover_name=columns[-1],
        hover_data=None if value_col is None else [value_col],
    )

    global_color_map = {}

    for level in columns:
        if level in color_map:  # Make sure the level has a color map
            for val, color in color_map[level].items():
                if str(val) not in global_color_map:
                    global_color_map[str(val)] = color

    for i, trace in enumerate(fig.data):
        # Initialize the 'colors' list for the trace if it's None
        if trace.marker.colors is None:
            trace.marker.colors = []

        # Initialize an empty list to store colors for each label
        colors = []

        # For each segment, apply the color based on its label
        for j, label in enumerate(trace.labels):
            color = global_color_map.get(
                str(label), "#000000"
            )  # Default to black if not found
            colors.append(color)

        # Assign the list of colors to the trace's marker colors
        trace.marker.colors = colors

    # Create dummy traces for the custom legend
    legend_entries = []
    for col, col_map in color_map.items():
        # Add feature label entry (i.e., 'Feature 1', 'Feature 2')
        legend_entries.append(
            go.Scatter(
                x=[None],
                y=[None],  # Empty points for the legend item
                mode="markers",
                marker=dict(
                    color="rgba(0,0,0,0)", size=10
                ),  # Invisible marker for the feature label
                name=f"{col}:",
                legendgroup=col,  # Group all values under the feature group
                showlegend=True,
                line=dict(
                    width=0
                ),  # Ensure the dummy trace doesn't create lines on the plot
            )
        )

        # Add value entries under the feature label
        for val, color in col_map.items():
            legend_entries.append(
                go.Scatter(
                    x=[None],
                    y=[None],  # Empty points for the legend item
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"{val}",
                    legendgroup=col,  # Group the values under the feature name
                    showlegend=True,
                    line=dict(
                        width=0
                    ),  # Ensure the dummy trace doesn't create lines on the plot
                )
            )

    # Add the legend entries to the layout
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        uniformtext=dict(minsize=10, mode="hide"),
        title={
            "text": title if title else f"Nested Visualization of {', '.join(columns)}",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        legend=dict(
            orientation="h",  # Horizontal layout for the legend
            yanchor="bottom",  # Align legend to the bottom
            y=-0.15,  # Position the legend below the chart
            x=+1.05,  # Position the legend below the chart
            tracegroupgap=10,  # Spacing between legend items
        ),
        # Set background color of plot and chart area to transparent
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        # Disable grid and axes lines
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),  # No grid or ticks on x-axis
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),  # No grid or ticks on y-axis
    )

    customdata = [
        " → ".join([f"{columns[i]}: {val}" for i, val in enumerate(i.split("/"))])
        for i in trace.ids
    ]

    # Pass the full path information as custom data
    fig.update_traces(
        branchvalues="total",
        customdata=customdata,  # Full path data as customdata
        hovertemplate="<b>Intersection</b>: %{customdata}<br><b>Percentage</b>: %{percentRoot:.1%}<extra></extra>",
        textinfo="label+percent root",
        insidetextorientation="radial",
        texttemplate="%{label}<br>%{percentRoot:.1%}",
    )

    # Add the legend entries for custom color patches
    for legend_entry in legend_entries:
        fig.add_trace(legend_entry)

    return fig


def plot_sampling_strategies(
    df, protected_attribute, target_column, width=1000, height=400
):
    """
    Create a visualization of class and group balances across different sampling strategies using Plotly.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    protected_attribute : str
        Column name of the protected attribute (e.g., 'sex')
    target_column : str
        Column name of the target class (e.g., 'income')
    width : int, optional
        Width of the figure, default is 1000
    height : int, optional
        Height of the figure, default is 400
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Create a copy of the dataframe
    data = df.copy()

    # Original names for later reference
    most_common_value = None

    # Convert to binary if not already
    if data[protected_attribute].nunique() > 2:
        # For demonstration purposes, convert to binary (most frequent vs rest)
        most_common = data[protected_attribute].value_counts().index[0]
        most_common_value = most_common
        data[protected_attribute] = (data[protected_attribute] == most_common).astype(
            int
        )

    if data[target_column].nunique() > 2:
        # For demonstration purposes, convert to binary (most frequent vs rest)
        most_common = data[target_column].value_counts().index[0]
        # Create clear mapping for target values
        target_values_map = {
            1: f"Most common: {most_common}",
            0: f"Other {target_column} values",
        }
        data[target_column] = (data[target_column] == most_common).astype(int)

    # Create subplot titles
    subplot_titles = [
        "Original data",
        "Class",
        "Class & Protected",
        "Protected",
        "Class (ratio)",
    ]

    # Create a figure with 5 subplots in a row
    fig = make_subplots(rows=1, cols=5, subplot_titles=subplot_titles)

    # First subplot: Original data distribution
    add_plot_distribution(fig, data, protected_attribute, target_column, 1, 1)

    # Second subplot: 'class' strategy
    class_data = apply_class_sampling(data, protected_attribute, target_column)
    add_plot_distribution(fig, class_data, protected_attribute, target_column, 1, 2)

    # Third subplot: 'class & protected' strategy
    class_protected_data = apply_class_protected_sampling(
        data, protected_attribute, target_column
    )
    add_plot_distribution(
        fig, class_protected_data, protected_attribute, target_column, 1, 3
    )

    # Fourth subplot: 'protected' strategy
    protected_data = apply_protected_sampling(data, protected_attribute, target_column)
    add_plot_distribution(fig, protected_data, protected_attribute, target_column, 1, 4)

    # Fifth subplot: 'class (ratio)' strategy
    class_ratio_data = apply_class_ratio_sampling(
        data, protected_attribute, target_column
    )
    add_plot_distribution(
        fig, class_ratio_data, protected_attribute, target_column, 1, 5
    )

    # Update subplot title with augmentation ratio
    for i, ax_title in enumerate(subplot_titles):
        if i > 0:  # Skip the first plot (original data)
            # Calculate augmentation ratio
            if i == 1:
                aug_data = class_data
            elif i == 2:
                aug_data = class_protected_data
            elif i == 3:
                aug_data = protected_data
            else:
                aug_data = class_ratio_data

            orig_size = len(data)
            aug_size = len(aug_data) - orig_size
            ratio = aug_size / len(aug_data)
            fig.layout.annotations[i].text = f"{ax_title}<br>r_aug={ratio:.2f}"

    # Update layout
    fig.update_layout(
        title_text=f"Sampling Strategies Comparison",
        title_x=0.5,
        width=width,
        height=height,
        showlegend=True,
        legend_title=target_column,
    )

    # Update all x-axes with the appropriate title
    x_title = protected_attribute

    for i in range(1, 6):
        fig.update_xaxes(title_text=x_title, row=1, col=i)

    # Rename x-axis tick values if non-binary protected attribute was used
    if most_common_value is not None:
        # Update x-axis tick values for all subplots
        for i in range(1, 6):
            fig.update_xaxes(
                tickvals=[0, 1], ticktext=["Other", most_common_value], row=1, col=i
            )

    # Update all y-axes (only first one with label to avoid redundancy)
    fig.update_yaxes(title_text="Count (Number of Samples)", row=1, col=1)

    return fig


def add_plot_distribution(fig, df, protected_attribute, target_column, row, col):
    """
    Add distribution plot to the specified subplot with counts on y-axis and percentages inside bars.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Calculate counts for each group
    counts_tab = pd.crosstab(df[protected_attribute], df[target_column])

    # Calculate percentages for text labels inside bars
    cross_tab_pct = (
        pd.crosstab(df[protected_attribute], df[target_column], normalize="index") * 100
    )

    # Get target class values and create a color mapping
    target_classes = sorted(df[target_column].unique())

    # Create a color mapping dictionary for target classes
    colors = [
        "rgba(31, 119, 180, 0.8)",
        "rgba(255, 127, 14, 0.8)",
        "rgba(44, 160, 44, 0.8)",
        "rgba(214, 39, 40, 0.8)",
        "rgba(148, 103, 189, 0.8)",
        "rgba(140, 86, 75, 0.8)",
    ]

    color_map = {
        target_class: colors[i % len(colors)]
        for i, target_class in enumerate(target_classes)
    }

    # Create stacked bar chart
    for i, target_class in enumerate(target_classes):
        show_legend = True if col == 1 else False

        # Calculate y offset for stacking based on counts
        y_offset = (
            counts_tab.loc[:, counts_tab.columns[:i]].sum(axis=1)
            if i > 0
            else pd.Series(0, index=counts_tab.index)
        )

        # Get color from mapping
        color = color_map[target_class]

        # Legend name
        legend_name = f"{target_column}={target_class}"

        # Text labels showing percentages inside bars
        text_labels = []
        for idx in counts_tab.index:
            pct = cross_tab_pct.loc[idx, target_class]
            text = f"{pct:.1f}%" if pct > 5 else ""
            text_labels.append(text)

        fig.add_trace(
            go.Bar(
                x=counts_tab.index,
                y=counts_tab[target_class],
                name=legend_name,
                offsetgroup=0,
                showlegend=show_legend,
                legendgroup=f"target_{target_class}",
                marker_color=color,
                text=text_labels,
                textposition="inside",
                hovertemplate=f"{protected_attribute}=%{{x}}<br>{legend_name}<br>Count=%{{y}}<br>Percentage=%{{text}}<extra></extra>",
                base=y_offset,
            ),
            row=row,
            col=col,
        )

    return fig


def apply_class_sampling(df, protected_attribute, target_column):
    """
    Separately for each group (0/1 in protected attribute) sample instances
    for the minority class to match the number in the majority class.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    result = df.copy()
    protected_values = df[protected_attribute].unique()

    for val in protected_values:
        group = df[df[protected_attribute] == val]
        class_counts = group[target_column].value_counts()

        if len(class_counts) < 2:
            continue

        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]

        # Generate synthetic samples for the minority class
        n_to_generate = n_majority - n_minority

        if n_to_generate <= 0:
            continue

        minority_samples = group[group[target_column] == minority_class]

        # Simple oversampling with replacement for demonstration
        synthetic_samples = minority_samples.sample(n_to_generate, replace=True)
        result = pd.concat([result, synthetic_samples])

    return result


def apply_class_protected_sampling(df, protected_attribute, target_column):
    """
    For the largest group, sample instances for the minority class to match
    the number in the majority class. For all other groups, sample for both classes
    to match the largest group.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    result = df.copy()

    # Find the largest group
    group_counts = df[protected_attribute].value_counts()
    largest_group = group_counts.idxmax()
    other_groups = [g for g in df[protected_attribute].unique() if g != largest_group]

    # Handle largest group: balance classes within it
    largest_group_data = df[df[protected_attribute] == largest_group]
    class_counts = largest_group_data[target_column].value_counts()

    if len(class_counts) >= 2:
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]

        # Generate synthetic samples for the minority class in largest group
        n_to_generate = n_majority - n_minority

        if n_to_generate > 0:
            minority_samples = largest_group_data[
                largest_group_data[target_column] == minority_class
            ]
            synthetic_samples = minority_samples.sample(n_to_generate, replace=True)
            result = pd.concat([result, synthetic_samples])

    # For each other group, sample both classes to match largest group's majority class size
    for group in other_groups:
        group_data = df[df[protected_attribute] == group]

        for class_val in df[target_column].unique():
            class_samples = group_data[group_data[target_column] == class_val]
            n_samples = len(class_samples)
            n_to_generate = n_majority - n_samples
            if n_samples == 0:
                raise Exception(
                    f"Group {group} has no members in prediction class {target_column}"
                )

            if n_to_generate > 0:
                synthetic_samples = class_samples.sample(n_to_generate, replace=True)
                result = pd.concat([result, synthetic_samples])

    return result


def apply_protected_sampling(df, protected_attribute, target_column):
    """
    Do not sample for the largest group, but only for all other groups
    to match the number in the largest group, without considering class labels.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    result = df.copy()

    # Find the largest group
    group_counts = df[protected_attribute].value_counts()
    largest_group = group_counts.idxmax()
    largest_group_size = group_counts[largest_group]
    other_groups = [g for g in df[protected_attribute].unique() if g != largest_group]

    # For each other group, sample to match the size of the largest group
    for group in other_groups:
        group_data = df[df[protected_attribute] == group]
        n_samples = len(group_data)
        n_to_generate = largest_group_size - n_samples

        if n_to_generate > 0:
            synthetic_samples = group_data.sample(n_to_generate, replace=True)
            result = pd.concat([result, synthetic_samples])

    return result


def apply_class_ratio_sampling(df, protected_attribute, target_column):
    """
    Do not sample for the largest group, but only for all other groups to match
    the class ratio of the largest group.

    This implementation follows the approach described in the prompt.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    result = df.copy()

    # Find the largest group
    group_counts = df[protected_attribute].value_counts()
    largest_group = group_counts.idxmax()
    largest_group_data = df[df[protected_attribute] == largest_group]

    # Calculate class percentages in largest group
    largest_group_class_counts = largest_group_data[target_column].value_counts()
    largest_group_total_count = len(largest_group_data)
    largest_group_class_percentages = (
        largest_group_class_counts / largest_group_total_count
    )

    # Process other groups
    other_groups = [g for g in df[protected_attribute].unique() if g != largest_group]

    for group in other_groups:
        group_data = df[df[protected_attribute] == group]
        group_class_counts = group_data[target_column].value_counts()
        group_total_count = len(group_data)

        # For each class in this group
        for class_label in df[target_column].unique():
            # If class exists in both largest group and current group
            if (
                class_label in largest_group_class_percentages
                and class_label in group_class_counts
            ):
                minority_class_percentage = (
                    group_class_counts[class_label] / group_total_count
                )
                max_length_df_class_percentage = largest_group_class_percentages[
                    class_label
                ]
                class_instances = group_class_counts[class_label]

                size = 0
                if max_length_df_class_percentage != minority_class_percentage:
                    # Calculate how many instances to add to match the ratio
                    additional_instances = (
                        max_length_df_class_percentage * group_total_count
                        - class_instances
                    ) / (1 - max_length_df_class_percentage)
                    size = int(additional_instances)

                if size > 0:
                    # Get samples of this class in this group
                    class_group_samples = group_data[
                        group_data[target_column] == class_label
                    ]

                    # Simple oversampling with replacement
                    synthetic_samples = class_group_samples.sample(size, replace=True)
                    result = pd.concat([result, synthetic_samples])

            # If class exists in largest group but not in current group
            elif (
                class_label in largest_group_class_percentages
                and class_label not in group_class_counts
            ):
                max_length_df_class_percentage = largest_group_class_percentages[
                    class_label
                ]

                # Calculate how many instances to add
                size = int(
                    max_length_df_class_percentage
                    * group_total_count
                    / (1 - max_length_df_class_percentage)
                )

                if size > 0 and class_label in df[target_column].value_counts():
                    # Get samples of this class from the whole dataset
                    class_samples = df[df[target_column] == class_label]

                    # Simple oversampling with replacement
                    synthetic_samples = class_samples.sample(size, replace=True)

                    # Set protected attribute to current group
                    synthetic_samples = synthetic_samples.copy()
                    synthetic_samples[protected_attribute] = group

                    result = pd.concat([result, synthetic_samples])

    return result


@metric(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=(
        "fairbench",
        "pandas",
        "plotly",
        "ucimlrepo",
        "pygrank",
    ),
)
def augmentation_report(
    dataset: Dataset,
    model: EmptyModel,
    sensitive: List[str],
) -> HTML:
    """
    <img src="https://raw.githubusercontent.com/arjunroyihrpa/MMM_fair/main/images/mmm-fair.png" alt="Based on MMM-Fair" style="float: left; margin-right: 5px; margin-bottom: 5px; height: 80px;"/>

    This module generates an interactive HTML report featuring a
    <a href="https://plotly.com/python/sunburst-charts/" target="_blank">Plotly sunburst pie chart</a> visualization
    to explore imbalances in the dataset based on subgroups defined by sensitive attributes and the target variable.
    This helps to quickly identify any potential imbalances in the dataset, allowing users to assess fairness and
    identify areas that may require intervention, such augmentation.

    Furthermore, the report visually presents different augmentation strategies designed to mitigate data imbalances,
    per sensitive attribute, as an interactive <a href="https://plotly.com/python/bar-charts/." target="_blank">Plotly bar chart</a>.
    These strategies adjust the distribution of the dataset by oversampling specific subgroups with synthetic samples,
    ensuring more equitable representation of sensitive attributes and target classes. The following
    augmentation strategies are visualized:

    - **Class:** Balances the class distribution within each subgroup by sampling the minority class.
    - **Class & Protected:** Ensures equal sample distribution across all subgroups by sampling both majority
      and minority classes.
    - **Protected:** Balances the number of instances across different groups without considering class labels.
    - **Class (Ratio):** Maintains the same class ratio across all groups as found in the largest group.

    These visualizations allow users to observe the effects of each strategy on the data distribution, helping to
    understand how the dataset is being augmented.

    For more information, refer to our full paper:
    **"Synthetic Tabular Data Generation for Class Imbalance and Fairness: A Comparative Study"**
    [Link to paper](https://arxiv.org/pdf/2409.05215).
    """
    import pandas as pd
    import numpy as np

    dataset = dataset.to_csv(sensitive)
    non_categorical = [col for col in sensitive if col not in dataset.cat]
    assert not non_categorical, (
        f"Non-categorical sensitive attributes not allowed in augmentation report: {non_categorical}. "
        f"Current categorical columns are: {dataset.cat}"
    )
    df = dataset.df
    target = (
        "1"
        if "1" in dataset.labels
        else (
            "yes"
            if "yes" in dataset.labels
            else dataset.labels[dataset.labels.__iter__().__next__()]
        )
    )
    df["class " + target] = pd.DataFrame(
        np.array(dataset.labels[target], dtype=float)
    ).values
    target = "class " + target
    notify_progress(1 / (1 + len(sensitive)), "Creating figures: sunburst chart")
    fig = generate_nested_pie_chart(df, [target] + sensitive)
    augmentation_html_plots = []
    for i, sens in enumerate(sensitive):
        notify_progress((1 + i) / (1 + len(sensitive)), f"Creating figures: {sens}")
        augmentation_html_plots.append(
            plot_sampling_strategies(df, sens, target).to_html(
                include_plotlyjs="cdn", full_html=False
            )
        )
    notify_progress(1, f"Converting to html")
    notify_end()

    fig.update_layout(autosize=True, height=600)
    main_html_content = fig.to_html(include_plotlyjs="cdn", full_html=False)

    faq_html = """
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

    complete_html = f"""
    <html>
    <head>
        <title>Augmentation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .description {{ margin-bottom: 20px; }}
            .plot-container {{ margin-bottom: 40px; }}

            /* Make the Attribute Distribution Overview section smaller */
            .overview-container {{
                width: 70vw; /* 50% of the viewport width */
                max-width: 1000px; 
                height: 600px; 
                margin: 0 auto 40px auto;
                padding: 15px;
                box-sizing: border-box;
            }}
            .model-comparison {{
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Augmentation Report</h1>
            <hr/>
            {faq_html}

            <div class="faq-container">
                <div class="faq-box">
                    <h3>❓ What is this?</h3>
                    <p>This report provides is generated by MAI-BIAS to generate an overview of dataset imbalances across 
                    the intersection of sensitive attributes and prediction targets using the MMM-Fair library.
                    Intersectionality emphasizes that people experience overlapping systems of discrimination 
                    based on multiple identity characteristics (race, gender, class, sexual orientation, 
                    disability, etc.). This is reflected also in how AI systems reproduce forms of discrimination. 
                    As an example of intersectional bias <b>[1]</b> race and 
                    gender together affect algorithmic performance of commercial facial-analysis systems;
                    worst performance for darker-skinned women demonstrates a compounded disparity 
                    that would be missed if the analysis looked only at race or only at gender. 
                    <br><br><b>[1]</b><i> Buolamwini, J., & Gebru, T. (2018, January). Gender shades: Intersectional accuracy disparities 
                    in commercial gender classification. In Conference on fairness, accountability and transparency 
                    (pp. 77-91). PMLR.</i>
                    </p>
                    <p>
                    <br>
                    <p>Imbalanced datasets can lead to biased model behavior, as underrepresented subgroups 
                    are more difficult to predict correctly. Intersections of many sensitive attributes (e.g., 
                    low-income hispanic woman) may create tiny or empty groups.
                    Augmentation strategies increase the representation
                    of such subgroups, which can improve fairness and robustness of downstream models.</p>
                </div>
        
                <div class="faq-box">
                      <h3>❗ Summary</h3>
                    <p>An interactive <a href="https://plotly.com/python/sunburst-charts/" target="_blank">sunburst chart</a>,
                    visualizes how subgroups form and how large or small they are compared to the total dataset.
                    This summarizes the distribution of data across sensitive attributes 
                    <i>{', '.join(sensitive)}</i> and the prediction target. 
                    Sensitive attributes are represented as concentric rings, where each segment corresponds 
                    to an intersectional subgroup. Hover over a segment to view its subgroup path and proportion in 
                    the dataset, and click on it to focus on the particular intersection.</p>
                    <br>
                    <p>This report also contains bar charts compare original and augmented distributions for each 
                    strategy, as well as references and research findings that you can consult. 
                    The annotation <i>r_aug</i> indicates the fraction of synthetic samples added to the dataset under 
                    that strategy.</p>
                </div>
            </div>
            <hr/>
            
            <h2>Attribute Distribution Overview</h2>
            <div class="description">
                <p>Including the prediction target in the inner disk, each ring represents a sensitive attribute.
                Segments then correspond to intersectional subgroups that occur by combining each attribute's values
                with (sub)groups of greater granularity. Click on an inner disk partition, the first ring, etc to progressively
                focus on subgroups of more intersections. Click on the inner disk if it focuses on a specific partition
                to go back a step. Hover over a segment to see the intersectional group it represents and the proportion
                of samples in the population contained in that group.
            </div>
            <div class="plot-container overview-container">
                {main_html_content}
            </div>
            
            <h2>Data Augmentation Strategies</h2>
            <p>Sampling strategies dictate the number of synthetic samples to generate from each subgroup, to create the final augmented dataset. 
            The following strategies are compared:</p>
            <ul>
                <li><strong>Class:</strong> Balances the class distribution within each group separately by sampling the minority class.</li>
                <li><strong>Class & Protected:</strong> Ensures equal sample distribution across all subgroups by sampling both majority and minority classes.</li>
                <li><strong>Protected:</strong> Balances the number of instances across different groups without considering class labels.</li>
                <li><strong>Class (Ratio):</strong> Maintains the same class ratio across all groups as found in the largest group.</li>
            </ul>
            <p>For more information, refer to our full paper: <a href="https://arxiv.org/pdf/2409.05215" target="_blank">"Synthetic Tabular Data Generation for Class Imbalance and Fairness: A Comparative Study"</a></p>

            <p>The following plots visualize the impact of these strategies on data distribution. In those,
            <em>r_aug</em> represents the percentage of synthetic samples in the final dataset, 
            providing insight into how much the dataset has been augmented.</p>

            {''.join([f"<div class'plot-container'><h3>Augmentation Strategies for sensitive attribute {sensitive[i]}</h3>{plot_html}</div>" 
                      for i, plot_html in enumerate(augmentation_html_plots)])}
            
            <div class="model-comparison">
                <h2>Generative Models for Oversampling</h2>
                <p>Our study compared five state-of-the-art generative methods for synthetic tabular data generation:</p>
                
                <ul>
                    <li><strong>SDV-GC [1]:</strong> Uses various continuous distributions to model features and a multivariate Gaussian Copula to estimate feature covariance.</li>
                    <li><strong>CTGAN [2]:</strong> Adapts GANs for tabular data with mode-specific normalization to overcome imbalances.</li>
                    <li><strong>TVAE [2]:</strong> Trains a Variational Autoencoder to learn a low-dimensional Gaussian latent space for sampling.</li>
                    <li><strong>CART [3]:</strong> A tree-based method for column-wise data generation that samples in the leaves, suitable for mixed data types.</li>
                    <li><strong>SMOTE-NC [4]:</strong> A non-parametric method that generates samples by interpolating between line segments connecting real instances.</li>
                </ul>

                <h3>Key Findings</h3>
                <p>The experiments across four real-world datasets (Adult, German credit, Dutch census, and Credit card clients) revealed that:</p>
                <ul>
                    <li>The non-parametric <strong>CART</strong> model emerged as the top performer in most cases, showing superior results for both utility and fairness metrics.</li>
                    <li>CART was significantly more computationally efficient than other methods.</li>
                    <li>The class (ratio) sampling strategy generally led to the best fairness metrics while maintaining high utility, as it (usually) requires fewer synthetic samples (lower r_aug) to achieve equal class ratios between different subgroups.</li>
                </ul>
                
                <h3>References</h3>
                    <p>[1] Patki, N., Wedge, R., Veeramachaneni, K.: The synthetic data vault. In: 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pp. 399-410. IEEE (2016)</p>
                    <p>[2] Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K.: Modeling tabular data using conditional gan. Advances in Neural Information Processing Systems 32 (2019)</p>
                    <p>[3] Reiter, J.P.: Using CART to generate partially synthetic public use microdata. Journal of Official Statistics 21(3), 441 (2005)</p>
                    <p>[4] Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research 16, 321-357 (2002)</p>
            </div>

        </div>
    </body>
    </html>
    """

    return HTML(complete_html)
