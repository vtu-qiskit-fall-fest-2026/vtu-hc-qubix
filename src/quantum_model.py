
import pandas as pd
import numpy as np
from pathlib import Path
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

ROOT = Path(__file__).resolve().parents[1]

# Load data and match labels by building_id
X_all = pd.read_csv(ROOT / "data" / "train_values.csv")
y_all = pd.read_csv(ROOT / "data" / "train_labels.csv")

data = X_all.merge(
    y_all[["building_id", "damage_grade"]],
    on="building_id",
    validate="one_to_one",
)

# Use four numeric features for a small quantum experiment
feature_cols = [
    "age",
    "area_percentage",
    "height_percentage",
    "count_floors_pre_eq",
]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = data["damage_grade"]

# Small stratified subset to keep quantum kernel runtime manageable
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=120,
    random_state=42,
    stratify=y,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.25,
    random_state=42,
    stratify=y_sample,
)

# Scale using training data only; reduce to four features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Keep angles in a suitable range for the feature map
X_train_q = np.pi * np.tanh(X_train_pca)
X_test_q = np.pi * np.tanh(X_test_pca)

print("Train samples:", len(X_train_q))
print("Test samples:", len(X_test_q))
print("Building 4-qubit ZZ feature map...")

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear",
)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

# Classical SVC on the same reduced data, for comparison
print("\nTraining classical RBF-SVM...")
classical_model = SVC(kernel="rbf")
classical_model.fit(X_train_q, y_train)
classical_pred = classical_model.predict(X_test_q)

print("Classical accuracy:", round(
    accuracy_score(y_test, classical_pred), 4
))
print("Classical balanced accuracy:", round(
    balanced_accuracy_score(y_test, classical_pred), 4
))
print("Classical macro F1:", round(
    f1_score(y_test, classical_pred, average="macro"), 4
))

# Quantum Support Vector Classifier
print("\nTraining quantum-kernel QSVC...")
start = time()

qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train_q, y_train)
quantum_pred = qsvc.predict(X_test_q)

print("Quantum kernel time (seconds):", round(time() - start, 2))
print("Quantum accuracy:", round(
    accuracy_score(y_test, quantum_pred), 4
))
print("Quantum balanced accuracy:", round(
    balanced_accuracy_score(y_test, quantum_pred), 4
))
print("Quantum macro F1:", round(
    f1_score(y_test, quantum_pred, average="macro"), 4
))

print("\nQuantum classification report:")
print(classification_report(
    y_test, quantum_pred, labels=[1, 2, 3], zero_division=0
))


# Save experiment predictions for analysis
results = pd.DataFrame({
    "actual_grade": y_test.to_numpy(),
    "classical_prediction": classical_pred,
    "quantum_prediction": quantum_pred,
})

results.to_csv(ROOT / "results" / "quantum_comparison.csv", index=False)

print("Saved predictions to results/quantum_comparison.csv")


print("Quantum experiment completed.")
print("Note: This is a small simulator-based experiment, not proof of quantum advantage.")
