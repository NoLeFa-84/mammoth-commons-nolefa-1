from typing import Literal
from mammoth_commons.models import SklearnValidator
from mammoth_commons.integration import loader


@loader(
    namespace="mammotheu", version="v054", python="3.13", packages=("scikit-learn",)
)
def model_from_sklearn(
    architecture: Literal[
        "Logistic regression", "SVC", "Decision tree", "tabicl"
    ] = "Logistic regression",
    fraction_of_training_set: float = 0.2,
    train_with_sensitive: bool = True,
) -> SklearnValidator:
    """
    <img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/list.png?raw=true"
    alt="dataset focus" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>assess an sklearn architecture (experimental)</h3>
    Checks whether biases can be discovered on an sklearn architecture. For simple architectures,
    a more in-depth analysis can be gleaned by selecting no model and performing an sklearn audit
    of the dataset.

    Args:
        architecture: The model's architecture.
        fraction_of_training_set: The fraction of data samples to be considered part of the training set.
        train_with_sensitive: Whether model training included the sensitive attributes that will be analysed in the next step or not. Including those attributes could help mitigate bias for some bias-aware training algorithms. Leave checked if you just trained the model with all available attributes.
    """
    fraction_of_training_set = float(fraction_of_training_set)

    if architecture == "Logistic regression":
        return SklearnValidator(
            """from sklearn.linear_model import LogisticRegression;commons = LogisticRegression(max_iter=1000)""",
            safe_imports=("sklearn",),
            require_install=("scikit-learn",),
            fraction_of_training_set=fraction_of_training_set,
            train_with_sensitive=train_with_sensitive,
        )

    if architecture == "SVC":
        return SklearnValidator(
            """from sklearn.svm import SVC;commons = SVC()""",
            safe_imports=("sklearn",),
            require_install=("scikit-learn",),
            fraction_of_training_set=fraction_of_training_set,
            train_with_sensitive=train_with_sensitive,
        )

    if architecture == "Decision tree":
        return SklearnValidator(
            """from sklearn.tree import DecisionTreeClassifier;commons = DecisionTreeClassifier(max_depth=5)""",
            safe_imports=("sklearn",),
            require_install=("scikit-learn",),
            fraction_of_training_set=fraction_of_training_set,
            train_with_sensitive=train_with_sensitive,
        )

    if architecture == "tabicl":
        return SklearnValidator(
            """from tabicl import TabICLClassifier;commons = TabICLClassifier()""",
            safe_imports=("tabicl",),
            require_install=("tabicl",),
            fraction_of_training_set=fraction_of_training_set,
            train_with_sensitive=train_with_sensitive,
        )
