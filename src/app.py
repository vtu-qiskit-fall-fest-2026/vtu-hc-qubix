
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FEATURES = [
    "age",
    "area_percentage",
    "height_percentage",
    "count_floors_pre_eq",
]

st.set_page_config(
    page_title="Q-RESQ | Building Damage Predictor",
    page_icon="🏚️",
    layout="wide",
)

@st.cache_resource
def train_demo_model():
    values_path = DATA_DIR / "train_values.csv"
    labels_path = DATA_DIR / "train_labels.csv"

    if not values_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Expected data/train_values.csv "
            "and data/train_labels.csv inside the project folder."
        )

    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)

    data = values.merge(
        labels[["building_id", "damage_grade"]],
        on="building_id",
        validate="one_to_one",
    )

    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = data["damage_grade"].astype(int)

    # Keep this demo training lightweight
    if len(data) > 1200:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=1200,
            random_state=42,
            stratify=y,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, test_accuracy


st.title("🏚️ Q-RESQ")
st.subheader("Earthquake Building Damage Prediction")
st.write(
    "Enter basic building details to see a demo prediction "
    "from a classical machine-learning model."
)

st.warning(
    "Research prototype only. This prediction is not a certified "
    "structural-safety assessment and must not be used to decide "
    "whether a building is safe to occupy."
)

try:
    model, test_accuracy = train_demo_model()

    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Building age (years)", min_value=0, max_value=200, value=20
        )
        area = st.number_input(
            "Area percentage", min_value=0, max_value=100, value=10
        )

    with right:
        height = st.number_input(
            "Height percentage", min_value=1, max_value=100, value=5
        )
        floors = st.number_input(
            "Number of floors before earthquake",
            min_value=1,
            max_value=100,
            value=3,
        )

    if st.button("Predict Damage Grade", type="primary"):
        input_data = pd.DataFrame([{
            "age": age,
            "area_percentage": area,
            "height_percentage": height,
            "count_floors_pre_eq": floors,
        }])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["classifier"].classes_

        st.markdown("### Prediction Result")

        descriptions = {
            1: "Grade 1 — lower damage category in the dataset",
            2: "Grade 2 — intermediate damage category in the dataset",
            3: "Grade 3 — higher damage category in the dataset",
        }

        st.metric("Predicted Damage Grade", prediction)
        st.write(descriptions.get(prediction, f"Grade {prediction}"))

        st.caption(
            "These grade descriptions are relative to the dataset's "
            "labels; they are not a real-world safety rating."
        )

        probability_data = pd.DataFrame({
            "Damage Grade": [f"Grade {int(c)}" for c in classes],
            "Model Probability": probabilities,
        })

        st.write("Model probability estimates")
        st.bar_chart(
            probability_data.set_index("Damage Grade")
        )

    st.divider()
    st.subheader("Model Evaluation")
    st.metric(
        "Demo model test accuracy",
        f"{test_accuracy:.1%}",
    )
    st.caption(
        "This accuracy comes from a separate classical-model split. "
        "It is not directly comparable with the 30-sample quantum "
        "experiment unless the same test split is used."
    )

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    if comparison_path.exists():
        st.subheader("Classical vs Quantum Experiment")
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True)

        if "Model" in comparison.columns and "Accuracy" in comparison.columns:
            st.write("Accuracy comparison")
            st.bar_chart(comparison.set_index("Model")["Accuracy"])

    st.caption(
        "Q-RESQ is an experimental prototype. The quantum models "
        "do not demonstrate quantum advantage in the current results."
    )

except FileNotFoundError as error:
    st.error(str(error))
except Exception as error:
    st.error(f"Unable to load the demo: {error}")
    st.info(
        "Check that the dataset files, Python packages, and project "
        "folder structure are correct."
    )
