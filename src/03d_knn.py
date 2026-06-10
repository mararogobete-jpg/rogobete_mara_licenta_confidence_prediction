# 03d_knn.py
# Model de clasificare: K-Nearest Neighbors.
#
# Scop:
# 1. Aplicăm KNN pentru clasificarea confidence_class în 3 clase:
#    0 = low confidence, 1 = medium confidence, 2 = high confidence.
# 2. Folosim StandardScaler deoarece KNN se bazează pe distanțe.
# 3. Alegem hiperparametrii prin GridSearchCV pe un eșantion de participanți.
# 4. Evaluăm modelul final pe tot datasetul prin GroupKFold pe uid_participant.
# 5. Salvăm metricile, matricea de confuzie și classification report.
#
# Notă:
# KNN este costisitor computațional pe seturi mari de date, deoarece compară
# observațiile noi cu observațiile din train. Din acest motiv, tuningul se face
# pe un eșantion de participanți, iar evaluarea finală se face pe tot datasetul.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

N_SPLITS = 3
RANDOM_STATE = 42

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["low_0", "medium_1", "high_2"]

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

# Pentru tuning, folosim doar un eșantion de participanți.
# Evaluarea finală rămâne pe tot datasetul.
MAX_TUNING_PARTICIPANTS = 1200


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


def make_knn_model(**params):
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_jobs=-1,
            **params
        ))
    ])

    return model


def adauga_model_comparatie(comparison_rows, path, model_name):
    if path.exists():
        df_summary = pd.read_csv(path)

        f1_macro_col = "f1_macro_mean"
        if f1_macro_col not in df_summary.columns:
            f1_macro_col = "macro_f1_mean"

        f1_weighted_col = "f1_weighted_mean"
        if f1_weighted_col not in df_summary.columns:
            f1_weighted_col = "weighted_f1_mean"

        comparison_rows.append({
            "model": model_name,
            "accuracy_mean": df_summary.loc[0, "accuracy_mean"],
            "balanced_accuracy_mean": df_summary.loc[0, "balanced_accuracy_mean"],
            "f1_macro_mean": df_summary.loc[0, f1_macro_col],
            "f1_weighted_mean": df_summary.loc[0, f1_weighted_col],
        })


# ------------------------------------------------------------
# 3. Încărcăm datasetul
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

print_section("Model 03d: K-Nearest Neighbors")
print("Dataset:", DATA_PATH)
print("Shape inițial:", df.shape)

required_cols = FEATURES + [TARGET, GROUP_COL]
missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Lipsesc coloanele necesare: {missing_cols}")

df = df.dropna(subset=required_cols).copy()
df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP_COL]

print("Shape după dropna:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())

print_section("Variabile folosite în model")
print("Target:", TARGET)
print("Predictori:", FEATURES)

print_section("Distribuția globală a targetului")
print(class_distribution(y))


# ------------------------------------------------------------
# 4. Eșantion de participanți pentru tuning
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

unique_participants = df[GROUP_COL].drop_duplicates().to_numpy()

if len(unique_participants) > MAX_TUNING_PARTICIPANTS:
    sampled_participants = rng.choice(
        unique_participants,
        size=MAX_TUNING_PARTICIPANTS,
        replace=False
    )

    df_tuning = df[df[GROUP_COL].isin(sampled_participants)].copy()

else:
    df_tuning = df.copy()

X_tuning = df_tuning[FEATURES]
y_tuning = df_tuning[TARGET]
groups_tuning = df_tuning[GROUP_COL]

print_section("Eșantion pentru tuning KNN")
print("MAX_TUNING_PARTICIPANTS:", MAX_TUNING_PARTICIPANTS)
print("Shape tuning:", df_tuning.shape)
print("Participanți tuning:", df_tuning[GROUP_COL].nunique())

print()
print("Distribuția targetului în eșantionul de tuning:")
print(class_distribution(y_tuning))


# ------------------------------------------------------------
# 5. GridSearchCV pentru KNN
# ------------------------------------------------------------
# n_neighbors = K
# weights:
#   - uniform: fiecare vecin votează la fel
#   - distance: vecinii mai apropiați au pondere mai mare
# metric:
#   - euclidean: distanța în linie dreaptă
#   - manhattan: suma diferențelor absolute

print_section("GridSearchCV KNN")

param_grid = {
    "classifier__n_neighbors": [5, 11, 21, 51, 101, 201],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}

n_combinations = 1
for values in param_grid.values():
    n_combinations *= len(values)

print("Număr combinații:", n_combinations)
print("CV: GroupKFold cu 3 folduri")
print("Scoring principal: macro F1")

gkf_tuning = GroupKFold(n_splits=N_SPLITS)

base_model = make_knn_model()

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
    cv=gkf_tuning,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
)

grid_search.fit(
    X_tuning,
    y_tuning,
    groups=groups_tuning
)

print_section("Best params KNN")
print(grid_search.best_params_)
print("Best CV macro F1:", grid_search.best_score_)

cv_results_df = pd.DataFrame(grid_search.cv_results_)

cv_results_df.to_csv(
    REPORTS_DIR / "03d_knn_grid_search_all_results.csv",
    index=False
)

best_params_df = pd.DataFrame([{
    "best_score_f1_macro": grid_search.best_score_,
    "n_tuning_rows": len(df_tuning),
    "n_tuning_participants": df_tuning[GROUP_COL].nunique(),
    **grid_search.best_params_,
    "features": ", ".join(FEATURES),
}])

best_params_df.to_csv(
    REPORTS_DIR / "03d_knn_grid_search_best_params.csv",
    index=False
)

top_grid = (
    cv_results_df
    .sort_values("mean_test_f1_macro", ascending=False)
    .head(20)
)

top_grid.to_csv(
    REPORTS_DIR / "03d_knn_grid_search_top_20.csv",
    index=False
)

print_section("Top 10 combinații KNN")
cols_to_show = [
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "mean_train_f1_macro",
    "std_train_f1_macro",
    "param_classifier__n_neighbors",
    "param_classifier__weights",
    "param_classifier__metric",
]

print(top_grid[cols_to_show].head(10))


# ------------------------------------------------------------
# 6. Evaluare finală cu GroupKFold pe tot datasetul
# ------------------------------------------------------------

print_section("Evaluare finală KNN cu GroupKFold")

best_params = {
    key.replace("classifier__", ""): value
    for key, value in grid_search.best_params_.items()
}

gkf = GroupKFold(n_splits=N_SPLITS)

fold_results = []
confusion_matrices = []

all_y_true = []
all_y_pred = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    model = make_knn_model(**best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS)
    confusion_matrices.append(cm)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_test_pred.tolist())

    row = {
        "fold": fold_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_participants": groups_train.nunique(),
        "n_test_participants": groups_test.nunique(),

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
    }

    fold_results.append(row)

    print(
        f"Fold {fold_idx}: "
        f"train={len(train_idx):,}, test={len(test_idx):,}, "
        f"train_participanți={groups_train.nunique()}, "
        f"test_participanți={groups_test.nunique()}, "
        f"test_accuracy={test_metrics['accuracy']:.4f}, "
        f"test_balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"test_macro_F1={test_metrics['f1_macro']:.4f}, "
        f"gap_macro_F1={row['gap_f1_macro']:.4f}"
    )


# ------------------------------------------------------------
# 7. Rezumat final
# ------------------------------------------------------------

results_df = pd.DataFrame(fold_results)

summary_df = pd.DataFrame([{
    "model": "KNeighborsClassifier",
    "n_splits": N_SPLITS,
    "features": ", ".join(FEATURES),

    "n_neighbors": best_params["n_neighbors"],
    "weights": best_params["weights"],
    "metric": best_params["metric"],

    "accuracy_mean": results_df["test_accuracy"].mean(),
    "accuracy_std": results_df["test_accuracy"].std(),

    "balanced_accuracy_mean": results_df["test_balanced_accuracy"].mean(),
    "balanced_accuracy_std": results_df["test_balanced_accuracy"].std(),

    "f1_macro_mean": results_df["test_f1_macro"].mean(),
    "f1_macro_std": results_df["test_f1_macro"].std(),

    "f1_weighted_mean": results_df["test_f1_weighted"].mean(),
    "f1_weighted_std": results_df["test_f1_weighted"].std(),

    "train_f1_macro_mean": results_df["train_f1_macro"].mean(),
    "test_f1_macro_mean": results_df["test_f1_macro"].mean(),
    "gap_f1_macro_mean": results_df["gap_f1_macro"].mean(),

    "train_accuracy_mean": results_df["train_accuracy"].mean(),
    "test_accuracy_mean": results_df["test_accuracy"].mean(),
    "gap_accuracy_mean": results_df["gap_accuracy"].mean(),
}])

print_section("Rezumat KNN")
print(summary_df)


# ------------------------------------------------------------
# 8. Verificare overfitting
# ------------------------------------------------------------

print_section("Verificare overfitting: train vs test")

overfit_cols = [
    "fold",
    "train_f1_macro",
    "test_f1_macro",
    "gap_f1_macro",
    "train_accuracy",
    "test_accuracy",
    "gap_accuracy",
]

print(results_df[overfit_cols])


# ------------------------------------------------------------
# 9. Matrice de confuzie totală
# ------------------------------------------------------------

cm_total = np.sum(confusion_matrices, axis=0)

cm_df = pd.DataFrame(
    cm_total,
    index=["true_low_0", "true_medium_1", "true_high_2"],
    columns=["pred_low_0", "pred_medium_1", "pred_high_2"]
)

print_section("Matrice de confuzie totală KNN")
print(cm_df)


# ------------------------------------------------------------
# 10. Classification report
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

print_section("Classification report KNN")
print(report_df)


# ------------------------------------------------------------
# 11. Comparație cu modelele anterioare
# ------------------------------------------------------------

comparison_rows = []

adauga_model_comparatie(
    comparison_rows,
    REPORTS_DIR / "03a_baseline_clasificare_summary.csv",
    "DummyClassifier_most_frequent"
)

adauga_model_comparatie(
    comparison_rows,
    REPORTS_DIR / "03b3_mnlogit_groupkfold_summary.csv",
    "MNLogit_GroupKFold"
)

adauga_model_comparatie(
    comparison_rows,
    REPORTS_DIR / "03c_decision_tree_summary.csv",
    "DecisionTreeClassifier"
)

comparison_rows.append({
    "model": "KNeighborsClassifier",
    "accuracy_mean": summary_df.loc[0, "accuracy_mean"],
    "balanced_accuracy_mean": summary_df.loc[0, "balanced_accuracy_mean"],
    "f1_macro_mean": summary_df.loc[0, "f1_macro_mean"],
    "f1_weighted_mean": summary_df.loc[0, "f1_weighted_mean"],
})

comparison_df = pd.DataFrame(comparison_rows)

print_section("Comparație modele")
print(comparison_df)


# ------------------------------------------------------------
# 12. Salvăm rezultatele
# ------------------------------------------------------------

results_df.to_csv(
    REPORTS_DIR / "03d_knn_folduri.csv",
    index=False
)

summary_df.to_csv(
    REPORTS_DIR / "03d_knn_summary.csv",
    index=False
)

cm_df.to_csv(
    REPORTS_DIR / "03d_knn_confusion_matrix.csv"
)

report_df.to_csv(
    REPORTS_DIR / "03d_knn_classification_report.csv"
)

comparison_df.to_csv(
    REPORTS_DIR / "03d_comparatie_modele_knn.csv",
    index=False
)
