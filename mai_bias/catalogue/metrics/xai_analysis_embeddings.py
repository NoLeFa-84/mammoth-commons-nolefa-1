from mammoth_commons.datasets import ImagePairs
from mammoth_commons.models.pytorch import Pytorch
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric

# install facex lib using: pip install facextool


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("torch", "torchvision", "timm", "facextool", "numpy"),
)
def facex_embeddings(
    dataset: ImagePairs,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = 1,
    target_layer: str = None,
) -> HTML:
    """
        <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/facex.png?raw=true"
        alt="facex" style="float: left; margin-right: 15px; height: 36px;"/>
        <h3>facial features contributing to image comparisons</h3>


        <a href="https://github.com/gsarridis/faceX">FaceX</a> is used to analyze
        19 key face regions, such as the eyes, nose, mouth,
        hair, and skin. Then it identifies where face verification models would focus
        on matching pairs of images.
        <details><summary><i>Technical approach at a glance.</i></summary>
        <img src="https://github.com/gsarridis/faceX/raw/main/images/facex.JPG" alt="Based on Facex" style="max-width: 600px;"/>
        </details>

        <details><summary><i>Why is this useful?</i></summary>
        Image embeddings are analyzed. In tasks like face recognition, models
        generate a feature vector (embedding) for each image. These embeddings capture the unique
        characteristics of the image and can be compared to determine how similar or different two images
        are. FaceX helps explain which parts of an image contribute to its similarity or difference to a
        reference embedding, allowing you to understand the model's focus on specific facial
        features during the comparison.

        The key idea is that FaceX analyzes the facial regions in the image that most
        influence how the model compares the reference embedding with the new image's embedding. Rather than
        providing an explanation for individual images in isolation, FaceX aggregates information across the
        dataset, offering insights into how different parts of the face, such as the eyes, mouth, or hair,
        contribute to the similarity or difference between the reference and the image being compared.

        FaceX works by using a "reference" image (e.g., identity image) and evaluating how other images
        (e.g., selfies) compare to it.
        It looks at how various facial regions of the new image align with the reference image in the
        feature space. The tool then highlights which facial regions are most influential in the comparison,
        showing what aspects of the image are similar to or
        different from the concept embedding.

        Overall, FaceX gives you a better understanding of how the model processes and compares facial
        features by highlighting the specific regions that influence the similarity or difference in feature
        embeddings. This is especially useful for improving transparency and identifying potential biases in
        how face verification models represent and compare faces.
        </details>

        <span class="alert alert-warning alert-dismissible fade show" role="alert"
            style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i> GPU
            access is recommended for large datasets. </span>

    Args:
        target_class: This variable determines what kind of comparison you want to explore. If you set the target class to 1, FaceX will investigate the regions in the image that are similar to the reference embedding (i.e., explanations on why the model consider the two images similar). If you set the target class to 0, FaceX will show the regions that are most different from the reference embedding (i.e., explanations on why the model consider the two images dissimilar).
        target_layer: This parameter lets you choose which part of the model's neural network you want to analyze. In simple terms, a model consists of multiple layers that process information at different levels. The target layer refers to the specific layer in the model that you want to explain. The explanation will show you which regions of the face are most important to that layer's decision-making process. For example, the deeper layers of the model may focus on more complex features like facial structure, while earlier layers might focus on simpler features like edges and textures. Typically, you should opt for the last layer producing the final embeddings.
    """
    from facex.component import run_embeddings_mammoth
    import matplotlib

    matplotlib.use("Agg")

    assert "," not in target_layer, "Only one model layer can be analysed"
    target_class = int(target_class)

    html = run_embeddings_mammoth(
        dataset, sensitive[0], target_class, model.model, target_layer
    )
    html = (
        """
        <h1>Image explanations</h1>
        <p>FaceX analysed 19 facial regions and accessories to provide explanations. In the two illustrations below,
        left are face regions and right are hat and glasses. Blue are the least important regions and red the most
        important ones that are taken into account. Based on the outputs, try to the question of “where does a model
        focus on?”. We also show high-impact patches to help understand “what visual features trigger its focus?”.</p>
        """
        + html
    )
    return HTML(html)
