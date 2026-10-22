
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


# --------------------------------------------------
# Q-RESQ: Variational Quantum Classifier (VQC)
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

np.random.seed(SEED)
algorithm_globals.random_seed = SEED


# 1. Load dataset and match labels by building_id
X_all = pd.read_csv(DATA_DIR / "train_values.csv")
y_all = pd.read_csv(DATA_DIR / "train_labels.csv")

data = X_all.merge(
    y_all[["building_id", "damage_grade"]],
    on="building_id",
    validate="one_to_one",
)

# 2. Select four numeric features
feature_cols = [
    "age",
    "area_percentage",
    "height_percentage",
    "count_floors_pre_eq",
]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = data["damage_grade"].astype(int)

# 3. Select a small stratified subset for simulation
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=120,
    random_state=SEED,
    stratify=y,
)

# Same fixed split settings for repeatable experiments
X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.25,
    random_state=SEED,
    stratify=y_sample,
)

# 4. Scale using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Reduce to four dimensions
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 6. Convert features to angles suitable for the feature map
X_train_q = np.pi * np.tanh(X_train_pca)
X_test_q = np.pi * np.tanh(X_test_pca)

print("Q-RESQ Variational Quantum Classifier")
print("Train samples:", len(X_train_q))
print("Test samples:", len(X_test_q))

# 7. Build quantum feature map and ansatz
feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear",
)

ansatz = real_amplitudes(
    num_qubits=4,
    reps=1,
    entanglement="linear",
)

# 8. Set deterministic initial parameters and sampler
rng = np.random.default_rng(SEED)
initial_point = rng.uniform(
    -np.pi,
    np.pi,
    size=ansatz.num_parameters,
)

sampler = StatevectorSampler(seed=SEED)
optimizer = COBYLA(maxiter=30)

# 9. Create and train VQC
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=initial_point,
)

print("\nTraining VQC...")
vqc.fit(X_train_q, y_train.to_numpy())

# 10. Predict test labels
vqc_pred = np.asarray(vqc.predict(X_test_q)).reshape(-1).astype(int)

# 11. Evaluate model
accuracy = accuracy_score(y_test, vqc_pred)
balanced_accuracy = balanced_accuracy_score(y_test, vqc_pred)
macro_f1 = f1_score(
    y_test,
    vqc_pred,
    average="macro",
    zero_division=0,
)

print("\nVQC Results")
print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        vqc_pred,
        labels=[1, 2, 3],
        zero_division=0,
    )
)

# 12. Save predictions for model comparison
results = pd.DataFrame({
    "actual_grade": y_test.to_numpy(),
    "vqc_prediction": vqc_pred,
})

output_file = RESULTS_DIR / "vqc_comparison.csv"
results.to_csv(output_file, index=False)

print("\nSaved predictions to:", output_file)
print(
    "\nNote: This is a small simulator-based experiment. "
    "It does not demonstrate quantum advantage."
)
