
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

# Load data
X = pd.read_csv(ROOT / "data" / "train_values.csv")
y_df = pd.read_csv(ROOT / "data" / "train_labels.csv")

# Match labels to buildings using building_id
data = X.merge(
    y_df[["building_id", "damage_grade"]],
    on="building_id",
    how="inner",
    validate="one_to_one",
)

y = data["damage_grade"]
X = data.drop(columns=["building_id", "damage_grade"])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
    ]
)

# Keep the same test split for future model comparisons
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("Training classical baseline...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro"), 4),
)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (grades 1, 2, 3):")
print(confusion_matrix(y_test, predictions, labels=[1, 2, 3]))

print("\nBaseline training completed.")
