# 03a_baseline_clasificare.py
# Baseline pentru clasificarea nivelului de confidence.
#
# Scop:
# 1. Citim datasetul final pentru clasificare.
# 2. Folosim GroupKFold cu 3 folduri, grupat după uid_participant.
# 3. Antrenăm un DummyClassifier care prezice clasa majoritară din train.
# 4. Calculăm metrici de clasificare:
#    - accuracy
#    - balanced accuracy
#    - macro F1
#    - weighted F1

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

DATA_PATH = PROCESSED_DIR / "dataset_main_classification.csv"

TARGET = "confidence_class"
GROUP_COL = "uid_participant"

N_SPLITS = 3

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

CLASS_LABELS = [0, 1, 2]


# ------------------------------------------------------------
# 2. Funcții simple
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def class_distribution(y):
    counts = pd.Series(y).value_counts().sort_index()

    rows = []

    for cls in CLASS_LABELS:
        n = counts.get(cls, 0)
        rows.append({
            "class": cls,
            "n": int(n),
            "percent": n / len(y)
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 3. Încărcăm datasetul
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Nu găsesc data/processed/dataset_main_classification.csv. "
        "Rulează mai întâi 02_preprocesare_dataset.py."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

print_section("Baseline clasificare: DummyClassifier")
print("Dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())


# ------------------------------------------------------------
# 4. Verificări minime
# ------------------------------------------------------------

required_cols = FEATURES + [TARGET, GROUP_COL]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Lipsesc coloanele necesare: {missing_cols}")

df = df.dropna(subset=required_cols).copy()

df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP_COL]

print_section("Distribuția globală a targetului")
global_dist = class_distribution(y)
print(global_dist)


# ------------------------------------------------------------
# 5. Cross-validation cu GroupKFold
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=N_SPLITS)

fold_results = []
fold_distributions = []
confusion_matrices = []

print_section("Rezultate pe folduri")

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Baseline: prezice mereu clasa majoritară din train
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    majority_class = int(baseline.classes_[np.argmax(baseline.class_prior_)])

    fold_results.append({
        "fold": fold_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_participants": groups_train.nunique(),
        "n_test_participants": groups_test.nunique(),
        "majority_class_train": majority_class,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    })

    # Distribuții clase train/test
    train_dist = class_distribution(y_train)
    train_dist["fold"] = fold_idx
    train_dist["set"] = "train"

    test_dist = class_distribution(y_test)
    test_dist["fold"] = fold_idx
    test_dist["set"] = "test"

    fold_distributions.append(train_dist)
    fold_distributions.append(test_dist)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    confusion_matrices.append(cm)

    print(
        f"Fold {fold_idx}: "
        f"train={len(train_idx):,}, test={len(test_idx):,}, "
        f"train_participanți={groups_train.nunique()}, "
        f"test_participanți={groups_test.nunique()}, "
        f"clasa_majoritară={majority_class}, "
        f"accuracy={acc:.4f}, "
        f"balanced_accuracy={bal_acc:.4f}, "
        f"macro_F1={f1_macro:.4f}, "
        f"weighted_F1={f1_weighted:.4f}"
    )


# ------------------------------------------------------------
# 6. Rezumat rezultate
# ------------------------------------------------------------

results_df = pd.DataFrame(fold_results)

summary_df = pd.DataFrame([{
    "model": "DummyClassifier_most_frequent",
    "n_splits": N_SPLITS,
    "accuracy_mean": results_df["accuracy"].mean(),
    "accuracy_std": results_df["accuracy"].std(),
    "balanced_accuracy_mean": results_df["balanced_accuracy"].mean(),
    "balanced_accuracy_std": results_df["balanced_accuracy"].std(),
    "f1_macro_mean": results_df["f1_macro"].mean(),
    "f1_macro_std": results_df["f1_macro"].std(),
    "f1_weighted_mean": results_df["f1_weighted"].mean(),
    "f1_weighted_std": results_df["f1_weighted"].std(),
}])

print_section("Rezumat baseline")
print(summary_df)


# ------------------------------------------------------------
# 7. Matrice de confuzie totală
# ------------------------------------------------------------

cm_total = np.sum(confusion_matrices, axis=0)

cm_df = pd.DataFrame(
    cm_total,
    index=["true_low_0", "true_medium_1", "true_high_2"],
    columns=["pred_low_0", "pred_medium_1", "pred_high_2"]
)

print_section("Matrice de confuzie totală")
print(cm_df)


# ------------------------------------------------------------
# 8. Salvăm rezultatele
# ------------------------------------------------------------

results_path = REPORTS_DIR / "03a_baseline_clasificare_folduri.csv"
summary_path = REPORTS_DIR / "03a_baseline_clasificare_summary.csv"
cm_path = REPORTS_DIR / "03a_baseline_clasificare_confusion_matrix.csv"
dist_path = REPORTS_DIR / "03a_distributii_clase_folduri.csv"

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
cm_df.to_csv(cm_path)

fold_distributions_df = pd.concat(fold_distributions, ignore_index=True)
fold_distributions_df.to_csv(dist_path, index=False)