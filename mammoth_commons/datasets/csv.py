from typing import Iterable
import importlib
from mammoth_commons.datasets.dataset import Dataset, Labels


def pdt(col, numeric: bool):
    pd = importlib.import_module("pandas")
    preprocessing = importlib.import_module("sklearn.preprocessing")
    if numeric:
        col = col.fillna(0)
        arr_2d = col.values.reshape(-1, 1)
        return pd.DataFrame(preprocessing.StandardScaler().fit_transform(arr_2d))
    col = col.fillna("missing")
    return pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(col))


def pd_features(
    df,
    num: list[str],
    cat: list[str],
    sens: list[str] | None = None,
    transform=lambda x, numeric: x,
):
    sens = set() if sens is None else set(sens)
    pd = importlib.import_module("pandas")
    dfs = [transform(df[col], True) for col in num if col not in sens]
    dfs += [transform(pd.get_dummies(df[col]), False) for col in cat if col not in sens]
    return pd.concat(dfs, axis=1).values


class CSV(Dataset):
    def __init__(
        self,
        df,
        num: list[str],
        cat: list[str],
        labels: str | dict | Iterable | None,
        sens: list[str] | None = None,
    ):
        pd = importlib.import_module("pandas")
        super().__init__(Labels(dict()))
        self.df = df
        self.num = num
        self.cat = cat
        self.cols = num + cat  # TODO: add variable to keep transformed col names too
        sens = set() if sens is None else set(sens)
        if isinstance(labels, str):
            sens.add(labels)
        self.feats = [col for col in self.cols if col not in sens]
        self.labels = (
            Labels(
                pd.get_dummies(df[labels]).to_dict(orient="list")
                if isinstance(labels, str)
                else (
                    pd.get_dummies(labels).to_dict(orient="list")
                    if isinstance(labels, pd.Series)
                    else (
                        labels
                        if isinstance(labels, dict)
                        else {"1": labels, "0": 1 - labels}
                    )
                )
            )
            if not isinstance(labels, Labels)
            else labels
        )

    def to_numpy(self, features: list[str] | None = None):
        assert (
            features
        ), "Internal error: misused to_numpy - a selection of features is required"
        assert (
            len(features) > 2
        ), "Internal error: misused to_numpy - a selection of at least two features is required"
        feats = set(features if features is not None else self.cols)
        return pd_features(
            self.df,
            [col for col in self.num if col not in feats],
            [col for col in self.cat if col not in feats],
        )

    def to_pred(self, exclude: list[str]):
        return pd_features(self.df, self.num, self.cat, exclude, transform=pdt)

    def to_csv(self, sensitive: list[str]):
        return self

    def to_pandas(self):
        raise NotImplemented

    def to_aif360(
        self,
        label_col: str,
        sensitive_cols: list[str],
        favorable_label=1,
        unfavorable_label=0,
    ) -> tuple:
        """
        Converts this dataset into a BinaryLabelDataset and returns:
        (BinaryLabelDataset, List of transformed sensitive column names)
        """
        import pandas as pd
        from aif360.datasets import BinaryLabelDataset
        from sklearn.preprocessing import LabelBinarizer

        df = self.df.copy()
        new_sensitive_cols = []

        # 1. One-hot encode non-sensitive, non-label categorical columns
        encode_cols = [
            c
            for c in df.columns
            if c not in sensitive_cols + [label_col]
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if encode_cols:
            df = pd.get_dummies(df, columns=encode_cols, drop_first=False)

        # 2. Convert label to binary
        if (
            not pd.api.types.is_numeric_dtype(df[label_col])
            or df[label_col].nunique() != 2
        ):
            lb = LabelBinarizer()
            y = lb.fit_transform(df[label_col])
            if y.shape[1] != 1:
                raise ValueError(f"Label column '{label_col}' must be binary.")
            df[label_col] = y.reshape(-1)

        # 3. Handle sensitive attributes
        for col in sensitive_cols:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2:
                new_sensitive_cols.append(col)
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                new_sensitive_cols.extend(dummies.columns.tolist())

        # 4. Create AIF360 dataset
        aif_dataset = BinaryLabelDataset(
            df=df,
            label_names=[label_col],
            protected_attribute_names=new_sensitive_cols,
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
        )

        return aif_dataset, new_sensitive_cols
