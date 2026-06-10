# 04_comparatie_finala_metrici.py
# Tabel final de metrici pentru modelele principale.
#
# Modele incluse:
# 1. DummyClassifier baseline
# 2. MNLogit statsmodels out-of-sample prin GroupKFold
# 3. Decision Tree
# 4. KNN
# 5. LinearSVC

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["low", "medium", "high"]

MODEL_CONFUSION_FILES = [
    ("DummyClassifier_most_frequent", "03a_baseline_clasificare_confusion_matrix.csv"),
    ("MNLogit_GroupKFold", "03b3_mnlogit_groupkfold_confusion_matrix.csv"),
    ("DecisionTreeClassifier", "03c_decision_tree_confusion_matrix.csv"),
    ("KNeighborsClassifier", "03d_knn_confusion_matrix.csv"),
    ("LinearSVC", "03e_svm_confusion_matrix.csv"),
]


# ------------------------------------------------------------
# 2. Funcții
# ------------------------------------------------------------

def compute_metrics_from_cm(cm):
    cm = cm.astype(float)

    total = cm.sum()
    tp = np.diag(cm)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    accuracy = tp.sum() / total

    precision_per_class = np.divide(
        tp,
        predicted,
        out=np.zeros_like(tp),
        where=predicted != 0
    )

    recall_per_class = np.divide(
        tp,
        support,
        out=np.zeros_like(tp),
        where=support != 0
    )

    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) != 0
    )

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    weighted_precision = np.average(precision_per_class, weights=support)
    weighted_recall = np.average(recall_per_class, weights=support)
    weighted_f1 = np.average(f1_per_class, weights=support)

    balanced_accuracy = macro_recall

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "support_low": int(support[0]),
        "support_medium": int(support[1]),
        "support_high": int(support[2]),
        "support_total": int(total),
    }

    for idx, class_name in enumerate(CLASS_NAMES):
        metrics[f"precision_{class_name}"] = precision_per_class[idx]
        metrics[f"recall_{class_name}"] = recall_per_class[idx]
        metrics[f"f1_{class_name}"] = f1_per_class[idx]

    return metrics


# ------------------------------------------------------------
# 3. Construim tabelul final
# ------------------------------------------------------------

rows = []

for model_name, file_name in MODEL_CONFUSION_FILES:
    path = REPORTS_DIR / file_name

    if not path.exists():
        print(f"WARNING: lipsește {file_name}. Modelul {model_name} este omis.")
        continue

    cm_df = pd.read_csv(path, index_col=0)
    cm = cm_df.values

    metrics = compute_metrics_from_cm(cm)

    rows.append({
        "model": model_name,
        **metrics,
    })


summary_df = pd.DataFrame(rows)

summary_df = summary_df.sort_values(
    "macro_f1",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 4. Salvare
# ------------------------------------------------------------

cols_main = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]

cols_by_class = [
    "model",
    "precision_low",
    "recall_low",
    "f1_low",
    "precision_medium",
    "recall_medium",
    "f1_medium",
    "precision_high",
    "recall_high",
    "f1_high",
]

main_path = REPORTS_DIR / "04_comparatie_finala_metrici_summary.csv"
class_path = REPORTS_DIR / "04_comparatie_finala_metrici_pe_clase.csv"

summary_df[cols_main].to_csv(main_path, index=False)
summary_df[cols_by_class].to_csv(class_path, index=False)

print()
print("=" * 60)
print("Tabel final - metrici principale")
print("=" * 60)
print(summary_df[cols_main])

print()
print("=" * 60)
print("Tabel final - performanță pe clase")
print("=" * 60)
print(summary_df[cols_by_class])

print()
print("Fișiere salvate:")
print(main_path)
print(class_path)