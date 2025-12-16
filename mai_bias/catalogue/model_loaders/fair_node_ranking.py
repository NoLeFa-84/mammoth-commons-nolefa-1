from mammoth_commons.models import NodeRanking
from mammoth_commons.integration import loader, Options
from typing import Literal


@loader(namespace="mammotheu", version="v054", python="3.13", packages=("pygrank",))
def model_fair_node_ranking(
    diffusion: float = 0.85,
    redistribution: Literal["none", "uniform", "original"] = "original",
) -> NodeRanking:
    """
    <img src="https://github.com/MKLab-ITI/pygrank/blob/master/docs/pygrank.png?raw=true"
    alt="pygrank" style="float: left; margin-right: 5px; height: 32px;"/>
    <h3>fairness-aware node ranking algorithm</h3>

    Constructs a node ranking algorithm that is a variation non-personalized PageRank.
    The base algorithm is often computes a notion of centrality/structural
    importance for each node in the graph, and employs a diffusion parameter in the range [0, 1).
    Find more details on how the algorithm works based on the following seminal paper:

    <i>Page, L. (1999). The PageRank citation ranking: Bringing order to the web. Technical Report.</i>

    The base node ranking algorithm is enriched by fairness-aware interventions implemented
    by the <a href="https://pygrank.readthedocs.io/en/latest/">pygrank</a> library. The latter
    may run on various computational backends, but `numpy` is selected due to its compatibility
    with a broad range of software and hardware. All implemented algorithms transfer node score
    mass from over-represented groups of nodes to those with lesser average mass using different
    strategies that determine the redistribution details. Fairness is imposed in terms of centrality
    scores achieving similar score mass between groups. The three available strategies are
    described here:

    <ul>
    <li>`none` does not employ any fairness intervention and runs the base algorithm.</li>
    <li>`uniform` applies a uniform rank redistribution strategy.</li>
    <li>`original` tries to preserve the order of original node ranks by distributing more score mass to those.</li>
    </ul>

    Args:
        diffusion: The diffusion parameters of the corresponding PageRank algorithm.
        redistribution: The redistribution strategy. Can be none, uniform or original.
    """
    import pygrank as pg

    assert redistribution in [
        "none",
        "original",
        "uniform",
    ], "Invalid node score redistribution strategy."
    diffusion = float(diffusion)
    assert diffusion >= 0, "The diffusion should be non-negative"
    assert diffusion < 1, "The diffusion should be <1"  # careful not to allow 1

    return NodeRanking(redistribution=redistribution)
