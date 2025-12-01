from mammoth_commons.integration import loader
from mammoth_commons.models.node_ranking import NodeRanking


@loader(namespace="mammotheu", version="v054", python="3.13")
def model_normal_ranking(
    path: str,
) -> NodeRanking:
    """
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png"
    alt="pygrank" style="float: left; margin-right: 5px; height: 32px;"/>
    <h3>node ranking algorithm</h3>
    Loads a graph node ranking algorithm defined by the
    <a href="https://pygrank.readthedocs.io/en/latest/">pygrank</a> library.
    Algorithms loaded this way are used in their non-personalized capacity,
    which means that they compute some notion of centrality/structural importance
    for each node in the graph.

    Args:
        path: A local path or url pointing to the model's specification, as exported by pygrank.
    """

    return NodeRanking(path)
