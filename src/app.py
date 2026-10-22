
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------
# Q-RESQ | Quantum ML Disaster Response Research Prototype
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FEATURES = [
    "age",
    "area_percentage",
    "height_percentage",
    "count_floors_pre_eq",
]

FEATURE_LABELS = {
    "age": "Building age (years)",
    "area_percentage": "Area percentage",
    "height_percentage": "Height percentage",
    "count_floors_pre_eq": "Number of floors before earthquake",
}

st.set_page_config(
    page_title="Q-RESQ | Disaster Intelligence",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Dark navy + cyan styling
# -------------------------

st.markdown(
    """
    <style>
    :root {
        --qresq-bg: #0B1220;
        --qresq-panel: #121F33;
        --qresq-cyan: #45D6E8;
        --qresq-text: #E8F0FA;
        --qresq-muted: #A5B4C8;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(28, 105, 145, 0.18),
                transparent 34%
            ),
            var(--qresq-bg);
        color: var(--qresq-text);
    }

    [data-testid="stSidebar"] {
        background: #0E192A;
        border-right: 1px solid #26364D;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.85);
    }

    h1, h2, h3 {
        color: #F3F7FC !important;
    }

    p, label, li {
        color: var(--qresq-text);
    }

    .qresq-eyebrow {
        color: var(--qresq-cyan);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .qresq-hero {
        background: linear-gradient(120deg, #142A46, #10283B);
        border: 1px solid #2B526B;
        border-radius: 18px;
        padding: 26px 28px;
        margin-bottom: 18px;
    }

    .qresq-hero h1 {
        font-size: 2.25rem;
        margin: 8px 0 6px 0;
    }

    .qresq-hero p {
        color: #C4D5E8;
        margin-bottom: 0;
    }

    .qresq-tag {
        display: inline-block;
        color: #77F0DA;
        background: rgba(40, 183, 159, 0.12);
        border: 1px solid rgba(91, 224, 201, 0.38);
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetric"] {
        background: #121F33;
        border: 1px solid #263B55;
        border-radius: 14px;
        padding: 16px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #B8C9DC;
    }

    div[data-testid="stMetricValue"] {
        color: #45D6E8;
    }

    div.stButton > button[kind="primary"] {
        background: #19BBD0;
        color: #071521;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 46px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #65E5F0;
        color: #071521;
        border: 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #263B55;
        border-radius: 10px;
        overflow: hidden;
    }

    .qresq-muted {
        color: #A5B4C8;
        font-size: 0.88rem;
    }

    .qresq-result {
        background: #12283B;
        border: 1px solid #2B657A;
        border-radius: 14px;
        padding: 18px 20px;
    }

    .qresq-result-label {
        color: #A5B4C8;
        font-size: 0.85rem;
    }

    .qresq-result-grade {
        color: #45D6E8;
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
    }

    hr {
        border-color: #26364D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Shared page components
# -------------------------

def render_hero():
    st.markdown(
        """
        <div class="qresq-hero">
            <span class="qresq-tag">RESEARCH PROTOTYPE</span>
            <div class="qresq-eyebrow" style="margin-top:16px;">
                Quantum machine learning research
            </div>
            <h1>Q-RESQ</h1>
            <p>
                Intelligent Disaster Severity Prediction &amp;
                Emergency Response Research
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_notice():
    st.warning(
        "Research prototype only. This prediction is not a certified "
        "structural-safety assessment and must not be used to decide "
        "whether a building is safe to occupy."
    )


# -------------------------
# Classical demo prediction
# -------------------------

@st.cache_resource
def train_demo_model():
    values_path = DATA_DIR / "train_values.csv"
    labels_path = DATA_DIR / "train_labels.csv"

    if not values_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Add train_values.csv and "
            "train_labels.csv to the project's data/ folder. "
            "See the README for dataset instructions."
        )

    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)

    required_values = ["building_id"] + FEATURES
    required_labels = ["building_id", "damage_grade"]

    missing_values = [
        col for col in required_values if col not in values.columns
    ]
    missing_labels = [
        col for col in required_labels if col not in labels.columns
    ]

    if missing_values or missing_labels:
        raise ValueError(
            f"Unexpected dataset columns. Missing from values: "
            f"{missing_values}; missing from labels: {missing_labels}."
        )

    data = values.merge(
        labels[["building_id", "damage_grade"]],
        on="building_id",
        validate="one_to_one",
    )

    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = data["damage_grade"].astype(int)

    if len(data) < 30 or y.nunique() < 2:
        raise ValueError(
            "The dataset does not contain enough examples or classes "
            "to train the demo model."
        )

    # Preserve the existing lightweight demo-training approach.
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

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, test_accuracy, len(X_train), len(X_test)


# -------------------------
# Results and comparison
# -------------------------

@st.cache_data
def load_model_comparison():
    path = RESULTS_DIR / "model_comparison.csv"

    if not path.exists():
        return None

    comparison = pd.read_csv(path)

    required = {
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Test Samples",
    }

    if not required.issubset(comparison.columns):
        return None

    return comparison


def render_comparison():
    st.subheader("Classical vs Quantum ML")
    st.caption(
        "Loaded from results/model_comparison.csv. "
        "These are saved experiment results, not live retraining."
    )

    comparison = load_model_comparison()

    if comparison is None:
        st.info(
            "Model comparison results are not available yet. "
            "Check that results/model_comparison.csv exists and "
            "contains Model, Accuracy, Balanced Accuracy, Macro F1, "
            "and Test Samples columns."
        )
        return

    display_data = comparison.copy()

    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
    ]

    for column in metric_columns:
        display_data[column] = pd.to_numeric(
            display_data[column], errors="coerce"
        )

    display_data["Test Samples"] = pd.to_numeric(
        display_data["Test Samples"], errors="coerce"
    )

    metric_cols = st.columns(3)

    for col, metric in zip(metric_cols, metric_columns):
        with col:
            valid = display_data.dropna(subset=[metric])

            if valid.empty:
                st.metric(metric, "N/A")
            else:
                best_idx = valid[metric].idxmax()
                best_row = valid.loc[best_idx]
                st.metric(
                    f"Best recorded {metric.lower()}",
                    f"{best_row[metric]:.1%}",
                    help=f"Model: {best_row['Model']}",
                )

    st.markdown("#### Evaluation metrics")

    formatted = display_data.copy()
    for column in metric_columns:
        formatted[column] = formatted[column].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
        )

    formatted["Test Samples"] = formatted["Test Samples"].map(
        lambda value: str(int(value)) if pd.notna(value) else "N/A"
    )

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Accuracy comparison")

    accuracy_chart = display_data[["Model", "Accuracy"]].dropna()

    if not accuracy_chart.empty:
        st.bar_chart(
            accuracy_chart.set_index("Model"),
            y="Accuracy",
            color="#45D6E8",
        )
        st.caption(
            "Accuracy is shown as a proportion. Comparisons are only "
            "fair when models use comparable data splits and evaluation "
            "procedures; check the test-sample counts and experiment setup."
        )

    st.info(
        "The current recorded experiments do not establish quantum "
        "advantage. Interpret the metrics with the small quantum "
        "evaluation sample in mind."
    )


# -------------------------
# Sidebar navigation
# -------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="qresq-eyebrow">Q-RESQ / QUBIX</div>
        <h2 style="margin-top:6px;">Navigation</h2>
        <p class="qresq-muted">
            Disaster intelligence research dashboard
        </p>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Building prediction",
            "Model comparison",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Project status**")
    st.markdown(
        '<span class="qresq-tag">EXPERIMENTAL</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Building damage-grade demo. Not a certified safety assessment."
    )


# -------------------------
# Overview
# -------------------------

if page == "Overview":
    render_hero()
    render_safety_notice()

    st.subheader("Disaster Intelligence Dashboard")
    st.write(
        "Explore the existing building-damage prediction demo and "
        "review the saved classical-versus-quantum model experiments."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input features", "4")
        st.caption("Building characteristics")

    with col2:
        st.metric("Damage classes", "1 / 2 / 3")
        st.caption("Dataset grade labels")

    with col3:
        st.metric("Experiment families", "Classical + Quantum")
        st.caption("Separate prediction and evaluation workflows")

    st.divider()

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("### Project workflow")
        st.markdown(
            """
            1. Enter four building characteristics.
            2. Run the classical RBF-SVM demo prediction.
            3. Inspect the model's class-probability estimates.
            4. Review the saved QSVC, VQC and classical comparison.
            """
        )

    with right:
        st.markdown("### Available sections")
        st.markdown(
            """
            - **Building prediction:** input form and results
            - **Model comparison:** saved evaluation metrics
            - **Methodology:** dataset, model scope and limitations
            """
        )

    st.caption(
        "This prototype currently predicts earthquake building damage "
        "grades. It does not yet implement geographic risk mapping, "
        "population analysis or emergency-resource forecasting."
    )


# -------------------------
# Building prediction
# -------------------------

elif page == "Building prediction":
    render_hero()
    render_safety_notice()

    st.subheader("Building Damage-Grade Prediction")
    st.write(
        "Enter the four building features used by the existing "
        "classical demo model."
    )

    try:
        model, test_accuracy, train_count, test_count = train_demo_model()

        with st.form("building_prediction_form"):
            st.markdown("#### Building characteristics")

            left, right = st.columns(2)

            with left:
                age = st.number_input(
                    "Building age (years)",
                    min_value=0,
                    max_value=200,
                    value=20,
                    step=1,
                    help="Recorded age of the building.",
                )

                area = st.number_input(
                    "Area percentage",
                    min_value=0,
                    max_value=100,
                    value=10,
                    step=1,
                    help="The dataset's area_percentage feature.",
                )

            with right:
                height = st.number_input(
                    "Height percentage",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    help="The dataset's height_percentage feature.",
                )

                floors = st.number_input(
                    "Number of floors before earthquake",
                    min_value=1,
                    max_value=100,
                    value=3,
                    step=1,
                    help="The dataset's count_floors_pre_eq feature.",
                )

            submitted = st.form_submit_button(
                "Predict Damage Grade",
                type="primary",
                use_container_width=True,
            )

        st.caption(
            f"Demo model evaluation split: {train_count:,} training "
            f"samples and {test_count:,} test samples."
        )

        if submitted:
            input_data = pd.DataFrame(
                [
                    {
                        "age": age,
                        "area_percentage": area,
                        "height_percentage": height,
                        "count_floors_pre_eq": floors,
                    }
                ]
            )

            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            classes = model.named_steps["classifier"].classes_

            descriptions = {
                1: "Grade 1 — lower damage category in this dataset",
                2: "Grade 2 — intermediate damage category in this dataset",
                3: "Grade 3 — higher damage category in this dataset",
            }

            st.divider()
            st.markdown("### Prediction result")

            result_col, metric_col = st.columns([1.2, 0.8])

            with result_col:
                st.markdown(
                    f"""
                    <div class="qresq-result">
                        <div class="qresq-result-label">
                            Predicted dataset label
                        </div>
                        <div class="qresq-result-grade">
                            Grade {prediction}
                        </div>
                        <div>{descriptions.get(prediction, "Predicted class")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_col:
                st.metric(
                    "Demo model test accuracy",
                    f"{test_accuracy:.1%}",
                )
                st.caption(
                    "Measured on the model's held-out classical test split; "
                    "not the confidence of this individual prediction."
                )

            st.markdown("#### Input summary")

            summary = pd.DataFrame(
                {
                    "Feature": [
                        FEATURE_LABELS[feature] for feature in FEATURES
                    ],
                    "Value": [
                        age,
                        area,
                        height,
                        floors,
                    ],
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Model probability estimates")

            probability_data = pd.DataFrame(
                {
                    "Damage Grade": [
                        f"Grade {int(class_label)}"
                        for class_label in classes
                    ],
                    "Model Probability": probabilities,
                }
            )

            st.bar_chart(
                probability_data.set_index("Damage Grade"),
                y="Model Probability",
                color="#45D6E8",
            )

            probability_display = probability_data.copy()
            probability_display["Model Probability"] = (
                probability_display["Model Probability"].map(
                    lambda value: f"{value:.1%}"
                )
            )

            st.dataframe(
                probability_display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "These are the model's probability estimates for dataset "
                "classes, not calibrated real-world damage probabilities "
                "or a structural-safety rating."
            )

            render_safety_notice()

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Unable to run the building prediction: {error}")
        st.info(
            "Check the dataset columns, Python packages and project "
            "folder structure. The original dataset files must be "
            "present in the data/ directory."
        )


# -------------------------
# Model comparison
# -------------------------

elif page == "Model comparison":
    render_hero()
    render_comparison()


# -------------------------
# Methodology and limitations
# -------------------------

elif page == "Methodology":
    render_hero()
    render_safety_notice()

    st.subheader("Methodology")

    st.markdown(
        """
        **Dataset**

        The existing application uses earthquake building records and
        their `damage_grade` labels. The demo prediction uses four
        numerical building features.

        **Classical prediction pipeline**

        1. Load `data/train_values.csv` and `data/train_labels.csv`.
        2. Match records using `building_id`.
        3. Handle missing values in the four selected features.
        4. Train a StandardScaler + RBF-SVM pipeline.
        5. Report held-out test accuracy and class-probability estimates.

        **Quantum model experiments**

        QSVC and VQC experiments are run by separate project scripts.
        Their saved comparison results are displayed from the
        `results/` directory. Opening this page does not retrain those
        quantum models.
        """
    )

    st.divider()
    st.subheader("Current scope and limitations")

    st.markdown(
        """
        - This app predicts dataset damage grades from four building features.
        - The classical demo model and quantum experiments may use
          different evaluation splits.
        - Quantum experiment metrics must be interpreted in light of
          their small test sample.
        - No quantum advantage has been established by the current results.
        - Geographic high-risk region mapping, population analysis and
          emergency-resource forecasting are not implemented in this app.
        """
    )

    st.markdown("#### Dataset setup")

    st.code(
        """
Q-RESQ/
├── data/
│   ├── train_values.csv
│   └── train_labels.csv
├── results/
│   ├── model_comparison.csv
│   ├── quantum_comparison.csv
│   └── vqc_comparison.csv
└── src/
    └── app.py
""".strip(),
        language="text",
    )

    st.markdown(
        "Dataset source: "
        "[Nepal Earthquake Damage Assessment dataset]"
        "(https://www.drivendata.org/competitions/57/nepal-earthquake/)"
    )

    st.caption(
        "This application is an experimental research prototype, "
        "not a certified structural inspection or emergency-management system."
    )


# -------------------------
# Footer
# -------------------------

st.divider()
st.markdown(
    """
    <p class="qresq-muted" style="text-align:center;">
        Q-RESQ · Team Qubix · Quantum ML Research Prototype
    </p>
    """,
    unsafe_allow_html=True,
)
