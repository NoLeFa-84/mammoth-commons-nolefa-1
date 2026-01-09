import os.path

from mammoth_commons.datasets import Graph
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v054", python="3.13", packages=("pygrank",))
def data_graph(
    path: str = "",
) -> Graph:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/graph.png?raw=true"
    alt="graph" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>graph</h3>
    Loads the edges of a graph organized as rows of a comma-delimited file.

    Args:
        path: A url from which to load the edges, or a pygrank dataset name to be automatically downloaded, preprocessed and loaded.
    """
    import pygrank as pg

    _, graph, communities = next(pg.load_datasets_multiple_communities([path]))
    return Graph(graph, communities)
