from mammoth_commons.integration import loader
from mammoth_commons.models.node_ranking import NodeRanking


@loader(namespace="mammotheu", version="v053", python="3.13")
def model_normal_ranking(
    path: str,
) -> NodeRanking:
    """Loads a graph node ranking algorithm defined by the
    <a href="https://pygrank.readthedocs.io/en/latest/">pygrank</a> library.
    Algorithms loaded this way are used in their non-personalized capacity,
    which means that they compute some notion of centrality/structural importance
    for each node in the graph.

    Args:
        path: A local path or url pointing to the model's specification, as exported by pygrank.
    """

    return NodeRanking(path)
