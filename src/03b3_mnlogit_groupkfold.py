# 03b3_mnlogit_groupkfold.py
# Evaluare out-of-sample pentru MNLogit statsmodels prin GroupKFold.
#
# Scop:
# - păstrăm MNLogit ca model statistic apropiat de curs/seminar;
# - evaluăm predictiv modelul pe participanți nevăzuți;
# - nu amestecăm metricile in-sample cu cele out-of-sample.

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


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
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["low_0", "medium_1", "high_2"]

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]


# ------------------------------------------------------------
# 2. Funcții ajutătoare
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def class_distribution(y):
    counts = pd.Series(y).value_counts().sort_index()

    rows = []
    for cls in CLASS_LABELS:
        n = counts.get(cls, 0)
        rows.append({
            "class": cls,
            "n": int(n),
            "percent": n / len(y),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 3. Încărcare date
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

required_cols = FEATURES + [TARGET, GROUP_COL]
missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Lipsesc coloanele necesare: {missing_cols}")

df = df.dropna(subset=required_cols).copy()
df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP_COL]

print_section("MNLogit out-of-sample cu GroupKFold")
print("Dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())

print_section("Predictori")
print(FEATURES)

print_section("Distribuția targetului")
print(class_distribution(y))


# ------------------------------------------------------------
# 4. GroupKFold
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=N_SPLITS)

fold_rows = []
confusion_matrices = []

all_y_true = []
all_y_pred = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    model = sm.MNLogit(y_train, X_train_const)

    result = model.fit(
        method="lbfgs",
        maxiter=200,
        disp=False
    )

    train_probs = np.asarray(result.predict(X_train_const))
    test_probs = np.asarray(result.predict(X_test_const))

    y_train_pred = np.array(CLASS_LABELS)[np.argmax(train_probs, axis=1)]
    y_test_pred = np.array(CLASS_LABELS)[np.argmax(test_probs, axis=1)]

    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS)
    confusion_matrices.append(cm)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_test_pred.tolist())

    fold_rows.append({
        "fold": fold_idx,

        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_participants": groups_train.nunique(),
        "n_test_participants": groups_test.nunique(),

        "converged": result.mle_retvals.get("converged", np.nan),

        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "gap_accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],

        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "gap_balanced_accuracy": (
            train_metrics["balanced_accuracy"]
            - test_metrics["balanced_accuracy"]
        ),

        "train_macro_precision": train_metrics["macro_precision"],
        "test_macro_precision": test_metrics["macro_precision"],
        "gap_macro_precision": (
            train_metrics["macro_precision"]
            - test_metrics["macro_precision"]
        ),

        "train_macro_recall": train_metrics["macro_recall"],
        "test_macro_recall": test_metrics["macro_recall"],
        "gap_macro_recall": (
            train_metrics["macro_recall"]
            - test_metrics["macro_recall"]
        ),

        "train_macro_f1": train_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "gap_macro_f1": (
            train_metrics["macro_f1"]
            - test_metrics["macro_f1"]
        ),

        "train_weighted_precision": train_metrics["weighted_precision"],
        "test_weighted_precision": test_metrics["weighted_precision"],

        "train_weighted_recall": train_metrics["weighted_recall"],
        "test_weighted_recall": test_metrics["weighted_recall"],

        "train_weighted_f1": train_metrics["weighted_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "gap_weighted_f1": (
            train_metrics["weighted_f1"]
            - test_metrics["weighted_f1"]
        ),
    })

    print(
        f"Fold {fold_idx}: "
        f"train={len(train_idx):,}, test={len(test_idx):,}, "
        f"train_participanți={groups_train.nunique()}, "
        f"test_participanți={groups_test.nunique()}, "
        f"test_accuracy={test_metrics['accuracy']:.4f}, "
        f"test_balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"test_macro_precision={test_metrics['macro_precision']:.4f}, "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}, "
        f"gap_macro_f1={fold_rows[-1]['gap_macro_f1']:.4f}"
    )


# ------------------------------------------------------------
# 5. Rezumat
# ------------------------------------------------------------

folds_df = pd.DataFrame(fold_rows)

summary_df = pd.DataFrame([{
    "model": "MNLogit_GroupKFold",
    "n_splits": N_SPLITS,
    "features": ", ".join(FEATURES),

    "accuracy_mean": folds_df["test_accuracy"].mean(),
    "accuracy_std": folds_df["test_accuracy"].std(),

    "balanced_accuracy_mean": folds_df["test_balanced_accuracy"].mean(),
    "balanced_accuracy_std": folds_df["test_balanced_accuracy"].std(),

    "macro_precision_mean": folds_df["test_macro_precision"].mean(),
    "macro_precision_std": folds_df["test_macro_precision"].std(),

    "macro_recall_mean": folds_df["test_macro_recall"].mean(),
    "macro_recall_std": folds_df["test_macro_recall"].std(),

    "macro_f1_mean": folds_df["test_macro_f1"].mean(),
    "macro_f1_std": folds_df["test_macro_f1"].std(),

    "f1_macro_mean": folds_df["test_macro_f1"].mean(),
    "f1_macro_std": folds_df["test_macro_f1"].std(),

    "weighted_precision_mean": folds_df["test_weighted_precision"].mean(),
    "weighted_precision_std": folds_df["test_weighted_precision"].std(),

    "weighted_recall_mean": folds_df["test_weighted_recall"].mean(),
    "weighted_recall_std": folds_df["test_weighted_recall"].std(),

    "weighted_f1_mean": folds_df["test_weighted_f1"].mean(),
    "weighted_f1_std": folds_df["test_weighted_f1"].std(),

    "f1_weighted_mean": folds_df["test_weighted_f1"].mean(),
    "f1_weighted_std": folds_df["test_weighted_f1"].std(),

    "train_macro_f1_mean": folds_df["train_macro_f1"].mean(),
    "test_macro_f1_mean": folds_df["test_macro_f1"].mean(),
    "gap_macro_f1_mean": folds_df["gap_macro_f1"].mean(),

    "train_accuracy_mean": folds_df["train_accuracy"].mean(),
    "test_accuracy_mean": folds_df["test_accuracy"].mean(),
    "gap_accuracy_mean": folds_df["gap_accuracy"].mean(),
}])

print_section("Rezumat MNLogit GroupKFold")
print(summary_df)

print_section("Verificare overfitting")
print(folds_df[[
    "fold",
    "train_macro_f1",
    "test_macro_f1",
    "gap_macro_f1",
    "train_accuracy",
    "test_accuracy",
    "gap_accuracy",
]])

cm_total = np.sum(confusion_matrices, axis=0)

cm_df = pd.DataFrame(
    cm_total,
    index=["true_low_0", "true_medium_1", "true_high_2"],
    columns=["pred_low_0", "pred_medium_1", "pred_high_2"]
)

print_section("Matrice de confuzie totală MNLogit GroupKFold")
print(cm_df)

report_df = pd.DataFrame(
    classification_report(
        all_y_true,
        all_y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
).T

print_section("Classification report MNLogit GroupKFold")
print(report_df)


# ------------------------------------------------------------
# 6. Salvare rezultate
# ------------------------------------------------------------

folds_df.to_csv(
    REPORTS_DIR / "03b3_mnlogit_groupkfold_folduri.csv",
    index=False
)

summary_df.to_csv(
    REPORTS_DIR / "03b3_mnlogit_groupkfold_summary.csv",
    index=False
)

cm_df.to_csv(
    REPORTS_DIR / "03b3_mnlogit_groupkfold_confusion_matrix.csv"
)

report_df.to_csv(
    REPORTS_DIR / "03b3_mnlogit_groupkfold_classification_report.csv"
)
