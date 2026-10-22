
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

classical_file = RESULTS / "quantum_comparison.csv"
vqc_file = RESULTS / "vqc_comparison.csv"

classical_data = pd.read_csv(classical_file)
vqc_data = pd.read_csv(vqc_file)

# Confirm that both quantum experiments used the same test labels.
if not classical_data["actual_grade"].reset_index(drop=True).equals(
    vqc_data["actual_grade"].reset_index(drop=True)
):
    raise ValueError(
        "Test labels differ. Rerun the models using the same test split."
    )

y_true = classical_data["actual_grade"]

models = {
    "Classical RBF-SVM": classical_data["classical_prediction"],
    "Quantum QSVC": classical_data["quantum_prediction"],
    "Quantum VQC": vqc_data["vqc_prediction"],
}

rows = []

for model_name, predictions in models.items():
    rows.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, predictions), 4),
        "Balanced Accuracy": round(
            balanced_accuracy_score(y_true, predictions), 4
        ),
        "Macro F1": round(
            f1_score(y_true, predictions, average="macro", zero_division=0),
            4,
        ),
        "Test Samples": len(y_true),
    })

comparison = pd.DataFrame(rows)
output_file = RESULTS / "model_comparison.csv"
comparison.to_csv(output_file, index=False)

print("\nQ-RESQ Model Comparison")
print(comparison.to_string(index=False))
print(f"\nSaved to: {output_file}")
