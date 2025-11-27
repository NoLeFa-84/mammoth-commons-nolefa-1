import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load
df = pd.read_csv("bank.csv", sep=";").dropna()

X = df.drop("y", axis=1)
y = (df["y"] == "yes").astype(int)

numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Convert categorical → numeric indices
ordinal = OrdinalEncoder()
X[categorical_features] = ordinal.fit_transform(X[categorical_features])

# Preprocess
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse=False),
            categorical_features,
        ),
    ]
)

clf = Pipeline(
    [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=200))]
)

clf.fit(X, y)

# Export as ONE input
initial_types = [("input", FloatTensorType([None, X.shape[1]]))]

onnx_model = convert_sklearn(clf, initial_types=initial_types)

with open("bank_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Exported bank_model.onnx with ONE input tensor")
