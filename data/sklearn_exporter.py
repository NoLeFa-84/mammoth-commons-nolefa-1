import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# --------------------
# Load dataset
# --------------------
df = pd.read_csv("bank.csv", sep=";")

# --------------------
# Drop rows with *any* missing values
# --------------------
df = df.dropna()

# Target
X = df.drop("y", axis=1)
y = (df["y"] == "yes").astype(int)

# Detect column groups
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

# --------------------
# Preprocessing WITHOUT IMPUTERS (needed for ONNX)
# --------------------
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=200))]
)

clf.fit(X, y)

# --------------------
# ONNX Export
# --------------------
initial_types = []
for col in numeric_features:
    initial_types.append((col, FloatTensorType([None, 1])))
for col in categorical_features:
    initial_types.append((col, StringTensorType([None, 1])))

onnx_model = convert_sklearn(clf, initial_types=initial_types)

with open("bank_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
