# 03c2_tuning_decision_tree.py
# Tuning Decision Tree cu GroupKFold.
#
# Scop:
# 1. Verificăm dacă scorul Decision Tree poate fi îmbunătățit prin hiperparametri.
# 2. Folosim GroupKFold pe uid_participant pentru a evita data leakage.
# 3. Optimizăm după macro F1, deoarece ne interesează toate clasele, nu doar clasa majoritară.
# 4. Comparăm train vs test pentru a verifica overfitting-ul.
#
# Notă:
# Acest script este de verificare/tuning.
# Nu înlocuiește automat 03c_decision_tree.py.
# Dacă rezultatul este mai bun și stabil, actualizăm ulterior 03c cu cei mai buni parametri.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
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

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["low_0", "medium_1", "high_2"]

N_SPLITS_OUTER = 3
N_SPLITS_INNER = 3

RANDOM_STATE = 42

# Dacă vrei să folosești toate nucleele, pune -1.
# Dacă laptopul se blochează, pune 1 sau 2.
N_JOBS = -1


# ------------------------------------------------------------
# 2. Funcții simple
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
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


# ------------------------------------------------------------
# 3. Încărcăm datasetul
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

print_section("Tuning Decision Tree")
print("Dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())

print_section("Distribuția targetului")
target_dist = y.value_counts().sort_index().reset_index()
target_dist.columns = ["class", "n"]
target_dist["percent"] = target_dist["n"] / target_dist["n"].sum()
print(target_dist)


# ------------------------------------------------------------
# 4. Grila de hiperparametri
# ------------------------------------------------------------
# Grila este mai amplă decât modelul inițial, dar nu absurd de mare.
# Dacă durează foarte mult, redu max_depth sau min_samples_leaf.

param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 7, 10, 12, 15, 20, None],
    "min_samples_leaf": [50, 100, 300, 500, 1000, 2000],
    "min_samples_split": [100, 500, 1000, 3000, 5000],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 1e-6, 1e-5, 1e-4],
}

print_section("Grilă hiperparametri")
n_combinations = 1
for key, values in param_grid.items():
    n_combinations *= len(values)
    print(f"{key}: {values}")

print()
print("Număr combinații:", n_combinations)
print("Scor optimizat: f1_macro")
print("Validare: nested GroupKFold, outer=3, inner=3")


# ------------------------------------------------------------
# 5. Nested GroupKFold
# ------------------------------------------------------------
# Outer fold = evaluare finală.
# Inner fold = alegerea hiperparametrilor.
#
# Avantaj:
# evităm să raportăm un scor prea optimist doar pentru că am ales parametrii
# pe aceleași folduri pe care evaluăm modelul.

outer_cv = GroupKFold(n_splits=N_SPLITS_OUTER)

fold_results = []
all_best_params = []
confusion_matrices = []

all_y_true = []
all_y_pred = []

all_cv_results = []

print_section("Rezultate pe folduri outer")

for fold_idx, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):

    print()
    print("=" * 60)
    print(f"OUTER FOLD {fold_idx}")
    print("=" * 60)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Train participanți:", groups_train.nunique())
    print("Test participanți:", groups_test.nunique())

    inner_cv = GroupKFold(n_splits=N_SPLITS_INNER)

    base_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring={
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        },
        refit="f1_macro",
        cv=inner_cv,
        n_jobs=N_JOBS,
        return_train_score=True,
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_inner_score = grid_search.best_score_

    print()
    print("Best params:")
    print(best_params)
    print("Best inner macro F1:", best_inner_score)

    # Salvăm toate rezultatele din inner CV pentru acest fold.
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df["outer_fold"] = fold_idx
    all_cv_results.append(cv_results_df)

    # Predicții train/test cu cel mai bun model ales în inner CV.
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS)
    confusion_matrices.append(cm)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_test_pred.tolist())

    # Feature importance pe modelul final al foldului.
    feature_importance = dict(zip(FEATURES, best_model.feature_importances_))

    row = {
        "fold": fold_idx,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_participants": groups_train.nunique(),
        "n_test_participants": groups_test.nunique(),

        "best_inner_f1_macro": best_inner_score,

        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "gap_accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],

        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "gap_balanced_accuracy": train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"],

        "train_f1_macro": train_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "gap_f1_macro": train_metrics["f1_macro"] - test_metrics["f1_macro"],

        "train_f1_weighted": train_metrics["f1_weighted"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "gap_f1_weighted": train_metrics["f1_weighted"] - test_metrics["f1_weighted"],

        "importance_accuracy": feature_importance["accuracy"],
        "importance_rt_dec_log_z": feature_importance["rt_dec_log_z"],
        "importance_rt_x_acc": feature_importance["rt_x_acc"],
        "importance_lag_rt_dec_log_z": feature_importance["lag_rt_dec_log_z"],
    }

    for param_name, param_value in best_params.items():
        row[f"best_{param_name}"] = param_value

    fold_results.append(row)

    all_best_params.append({
        "fold": fold_idx,
        **best_params,
        "best_inner_f1_macro": best_inner_score,
        "test_f1_macro": test_metrics["f1_macro"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_weighted": test_metrics["f1_weighted"],
    })

    print()
    print("Metrici outer test:")
    print(f"accuracy = {test_metrics['accuracy']:.4f}")
    print(f"balanced_accuracy = {test_metrics['balanced_accuracy']:.4f}")
    print(f"macro_F1 = {test_metrics['f1_macro']:.4f}")
    print(f"weighted_F1 = {test_metrics['f1_weighted']:.4f}")
    print(f"gap_macro_F1 = {row['gap_f1_macro']:.4f}")


# ------------------------------------------------------------
# 6. Rezumat final tuning
# ------------------------------------------------------------

fold_results_df = pd.DataFrame(fold_results)
best_params_df = pd.DataFrame(all_best_params)

summary_df = pd.DataFrame([{
    "model": "DecisionTreeClassifier_tuned_nested_cv",
    "n_splits_outer": N_SPLITS_OUTER,
    "n_splits_inner": N_SPLITS_INNER,
    "scoring_refit": "f1_macro",
    "n_param_combinations": n_combinations,

    "accuracy_mean": fold_results_df["test_accuracy"].mean(),
    "accuracy_std": fold_results_df["test_accuracy"].std(),

    "balanced_accuracy_mean": fold_results_df["test_balanced_accuracy"].mean(),
    "balanced_accuracy_std": fold_results_df["test_balanced_accuracy"].std(),

    "f1_macro_mean": fold_results_df["test_f1_macro"].mean(),
    "f1_macro_std": fold_results_df["test_f1_macro"].std(),

    "f1_weighted_mean": fold_results_df["test_f1_weighted"].mean(),
    "f1_weighted_std": fold_results_df["test_f1_weighted"].std(),

    "train_f1_macro_mean": fold_results_df["train_f1_macro"].mean(),
    "test_f1_macro_mean": fold_results_df["test_f1_macro"].mean(),
    "gap_f1_macro_mean": fold_results_df["gap_f1_macro"].mean(),
}])

print_section("Rezumat tuning Decision Tree")
print(summary_df)

print_section("Best params pe folduri")
print(best_params_df)

print_section("Verificare overfitting train vs test")
print(fold_results_df[[
    "fold",
    "train_f1_macro",
    "test_f1_macro",
    "gap_f1_macro",
    "train_accuracy",
    "test_accuracy",
    "gap_accuracy",
]])


# ------------------------------------------------------------
# 7. Matrice de confuzie totală
# ------------------------------------------------------------

cm_total = np.sum(confusion_matrices, axis=0)

cm_df = pd.DataFrame(
    cm_total,
    index=["true_low_0", "true_medium_1", "true_high_2"],
    columns=["pred_low_0", "pred_medium_1", "pred_high_2"]
)

print_section("Matrice de confuzie totală - tuned Decision Tree")
print(cm_df)


# ------------------------------------------------------------
# 8. Classification report agregat
# ------------------------------------------------------------

report_dict = classification_report(
    all_y_true,
    all_y_pred,
    labels=CLASS_LABELS,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).T

print_section("Classification report - tuned Decision Tree")
print(report_df)


# ------------------------------------------------------------
# 9. Feature importance medie
# ------------------------------------------------------------

importance_cols = [
    "importance_accuracy",
    "importance_rt_dec_log_z",
    "importance_rt_x_acc",
    "importance_lag_rt_dec_log_z",
]

importance_summary = (
    fold_results_df[importance_cols]
    .mean()
    .reset_index()
)

importance_summary.columns = ["feature_raw", "importance_mean"]
importance_summary["feature"] = (
    importance_summary["feature_raw"]
    .str.replace("importance_", "", regex=False)
)

importance_summary = importance_summary[["feature", "importance_mean"]]
importance_summary = importance_summary.sort_values(
    "importance_mean",
    ascending=False
)

print_section("Importanța medie a variabilelor - tuned Decision Tree")
print(importance_summary)


# ------------------------------------------------------------
# 10. Top combinații din inner CV
# ------------------------------------------------------------

cv_results_all_df = pd.concat(all_cv_results, ignore_index=True)

top_cv_results = (
    cv_results_all_df
    .sort_values("mean_test_f1_macro", ascending=False)
    .head(30)
)

print_section("Top 30 combinații hiperparametri după inner CV")
cols_to_show = [
    "outer_fold",
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "mean_train_f1_macro",
    "std_train_f1_macro",
    "param_criterion",
    "param_max_depth",
    "param_min_samples_leaf",
    "param_min_samples_split",
    "param_class_weight",
    "param_ccp_alpha",
]

print(top_cv_results[cols_to_show])


# ------------------------------------------------------------
# 11. Salvăm rezultatele
# ------------------------------------------------------------

fold_results_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_folduri.csv",
    index=False
)

summary_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_summary.csv",
    index=False
)

best_params_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_best_params.csv",
    index=False
)

cm_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_confusion_matrix.csv"
)

report_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_classification_report.csv"
)

importance_summary.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_feature_importance.csv",
    index=False
)

cv_results_all_df.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_all_cv_results.csv",
    index=False
)

top_cv_results.to_csv(
    REPORTS_DIR / "03c2_tuning_decision_tree_top_30.csv",
    index=False
)


# ------------------------------------------------------------
# 12. Concluzie automată
# ------------------------------------------------------------

print_section("Concluzie 03c2")

print(
    "Tuningul Decision Tree a fost realizat prin nested GroupKFold: "
    "foldurile outer evaluează performanța finală, iar foldurile inner aleg "
    "hiperparametrii. Metrica optimizată este macro F1, deoarece toate clasele "
    "de confidence sunt relevante, inclusiv clasa medium."
)

print()
print("Fișiere salvate:")
print(" - reports/03c2_tuning_decision_tree_folduri.csv")
print(" - reports/03c2_tuning_decision_tree_summary.csv")
print(" - reports/03c2_tuning_decision_tree_best_params.csv")
print(" - reports/03c2_tuning_decision_tree_confusion_matrix.csv")
print(" - reports/03c2_tuning_decision_tree_classification_report.csv")
print(" - reports/03c2_tuning_decision_tree_feature_importance.csv")
print(" - reports/03c2_tuning_decision_tree_all_cv_results.csv")
print(" - reports/03c2_tuning_decision_tree_top_30.csv")