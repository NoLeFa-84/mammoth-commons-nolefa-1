import numpy as np
from mammoth_commons.datasets.dataset import Labels
from sklearn.model_selection import train_test_split
from mammoth_commons.datasets.csv import CSV
from mammoth_commons.externals import align_predictions, safeexec
from mammoth_commons.integration import install_package
from mammoth_commons.models.predictor import Predictor


class SklearnValidator(Predictor):
    def __init__(
        self,
        model,
        safe_imports: tuple = (),
        fraction_of_training_set: float = 0.2,
        random_state=None,
        train_with_sensitive=True,
        require_install: tuple[str] = ("scikit-learn",),
        architecture_variable: str = "commons",
    ):
        super().__init__()
        self.model = model
        self.split_ratio = fraction_of_training_set
        self.random_state = random_state
        self.train_with_sensitive = train_with_sensitive
        self.require_install = require_install
        self.safe_imports = safe_imports
        self.architecture_variable = architecture_variable

    def _labels_to_numpy(self, lbls: Labels) -> np.ndarray:
        cols = [np.asarray(v) for _, v in lbls.items()]
        assert cols, "Dataset contains no label columns"
        return np.column_stack(cols)

    def predict(self, dataset, sensitive):
        model = self.model
        if isinstance(model, str):
            for package in self.require_install:
                install_package(package)
            model = safeexec(
                model, self.architecture_variable, whitelist=self.safe_imports
            )
        dataset: CSV = dataset.to_csv(sensitive)
        X = dataset.to_pred([] if self.train_with_sensitive else sensitive)
        y_one_hot = self._labels_to_numpy(dataset.labels)
        n_samples = X.shape[0]
        all_idx = np.arange(n_samples)
        stratify = None
        if y_one_hot.shape[1] > 1 and len(np.unique(y_one_hot, axis=0)) > 1:
            stratify = np.argmax(y_one_hot, axis=1)
            y_one_hot = (
                stratify  # may adjust this so leaving stratification logic intact
            )
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=self.split_ratio,
            random_state=self.random_state,
            stratify=stratify,
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_one_hot[train_idx], y_one_hot[test_idx]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_labels_dict = {
            k: np.asarray(v)[test_idx].tolist() for k, v in dataset.labels.items()
        }
        test_labels = Labels(test_labels_dict)
        # WE ARE DELIBERATELY MODIFYING THE DATASET BELOW
        # BECAUSE THIS IS EASIER TO DO THAN NOTIFYING EVERY METHOD
        # OF THE CHANGES - IDEALLY WE WOULD RETURN A NEW DATASET ALONGSIDE
        # THE PREDICTIONS BUT THIS IS A TEMPORARY SOLUTIONS
        # THAT IS FINE TO BE KEPT AS PERMANENT ONE IF THERE
        # IS NOT ADDITIONAL INTEREST
        dataset.df = dataset.df.iloc[test_idx].reset_index(drop=True)
        return align_predictions(predictions, test_labels)
