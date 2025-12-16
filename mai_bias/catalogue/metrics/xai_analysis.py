from mammoth_commons.datasets import Image
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
def facex_regions(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = 1,
    target_layer: str = None,
) -> HTML:
    """
        <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/facex.png?raw=true"
        alt="facex" style="float: left; margin-right: 15px; height: 36px;"/>
        <h3>facial features contributing to image categorization</h3>

        <a href="https://github.com/gsarridis/faceX">FaceX</a> is used to analyze
        19 key face regions, such as the eyes, nose, mouth,
        hair, and skin. Then it identifies which parts of the face the model focuses on when making
        predictions about attributes like age, gender, or race.

        <span class="alert alert-warning alert-dismissible fade show" role="alert"
            style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i> GPU
            access is recommended for large datasets. </span>

        <details><summary><i>Technical approach at a glance.</i></summary>
        <img src="https://github.com/gsarridis/faceX/raw/main/images/facex.JPG" alt="Based on Facex" style="max-width: 600px;"/>
        </details>

        <details><summary><i>Why is this useful?</i></summary>

        Rather than explaining each individual image separately, FaceX aggregates information across the
        entire dataset, offering a broader view of the model's behavior. It looks at how the model activates
        different regions of the face for each decision. This aggregated information helps
        you see which facial features are most influential in the model's predictions, and whether certain
        features are being emphasized more than others.

        In addition to providing an overall picture of which regions are important, FaceX also zooms in on
        specific areas of the face - such as a section of the skin or a part of the hair - showing which
        patches of the image have the highest impact on the model's decision. This makes it easier to
        identify potential biases or problems in how the model is interpreting the face.

        Overall, with FaceX, you can quickly and easily get a better understanding of your model's
        decision-making process. This is especially useful for ensuring that your model is fair and
        transparent, and for spotting any potential biases that may
        affect its performance.
        </details>


    Args:
        target_class: This parameter allows you to specify which class you want to analyze. For example, if you are using the model to classify faces by gender, setting the target class to male (i.e., the integer identifier for males class) will show you how the model makes decisions about male faces.
        target_layer: This parameter lets you choose which part of the model's neural network you want to analyze. In simple terms, a model consists of multiple layers that process information at different levels. The target layer refers to the specific layer in the model that you want to explain. The explanation will show you which regions of the face are most important to that layer's decision-making process. For example, the deeper layers of the model may focus on more complex features like facial structure, while earlier layers might focus on simpler features like edges and textures. Typically, you should opt for the last layer prior to the classification layer.
    """
    from facex.component import run_mammoth
    import matplotlib

    matplotlib.use("Agg")

    assert "," not in target_layer, "Only one model layer can be analysed"
    target_class = int(target_class)
    html = run_mammoth(dataset, sensitive[0], target_class, model.model, target_layer)

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

    <div class="faq-container">
        <div class="faq-box">
            <h3>❓ What is this?</h3>
            <p>This collection of visual explanations is produced by MAI-BIAS using 
            the <a href="https://github.com/gsarridis/faceX" target="_blank">FaceX</a> library
            to explain how face attribute classifiers make decisions. It evaluates 19 facial regions 
            such as eyes, nose, mouth, hair, and skin, showing which areas influence predictions the most.</p>
            <br/>
            <p>Rather than analyzing images one by one, FaceX aggregates activations across the dataset 
            to reveal common patterns. It highlights high-impact regions and patches, helping identify 
            potential biases and ensuring greater transparency in how the model interprets faces.</p>
        </div>

        <div class="faq-box">
            <h3>❗ Summary</h3>
            <p>FaceX produces heatmaps where <span style="color:blue;">blue</span> marks less important 
            regions and <span style="color:red;">red</span> marks highly influential regions. 
            These maps answer the question: <i>"Where does the model focus?"</i>. 
            High-impact patches provide further detail on <i>"What visual features trigger this focus?"</i>.</p>
            <br/>
            <p>By combining regional importance with patch-level analysis, the report helps spot 
            possible biases in model reasoning — for example, whether the classifier over-relies 
            on irrelevant features like accessories instead of actual facial attributes.</p>
        </div>
    </div>
    """

    html = f"""
    <h1>Image explanations</h1>
    <hr/>
    {faq_html}
    <hr/>
    {html}
    """

    return HTML(html)
