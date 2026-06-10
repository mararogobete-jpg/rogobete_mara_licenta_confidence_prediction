# 03c_decision_tree.py
# Model de clasificare: Decision Tree.
#
# Scop:
# 1. Citim datasetul final pentru clasificare.
# 2. Folosim GroupKFold cu 3 folduri, pe uid_participant.
# 3. Antrenăm un DecisionTreeClassifier.
# 4. Evaluăm modelul prin accuracy, balanced accuracy, macro F1 și weighted F1.
# 5. Comparăm rezultatele cu baseline-ul și cu MNLogit out-of-sample.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

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

df = pd.read_csv(DATA_PATH, low_memory=False)

print_section("Model 03c: Decision Tree")
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

print_section("Variabile folosite în model")
print("Target:", TARGET)
print("Predictori:", FEATURES)

print_section("Distribuția globală a targetului")
print(class_distribution(y))


# ------------------------------------------------------------
# 5. Definim modelul
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=50,
    min_samples_split=100,
    class_weight="balanced",
    ccp_alpha=0.0,
    random_state=42
)


# ------------------------------------------------------------
# 6. Cross-validation cu GroupKFold
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=N_SPLITS)

fold_results = []
confusion_matrices = []
importance_rows = []

print_section("Rezultate pe folduri")

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    fold_results.append({
        "fold": fold_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_participants": groups_train.nunique(),
        "n_test_participants": groups_test.nunique(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    })

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    confusion_matrices.append(cm)

    for feature_name, importance_value in zip(FEATURES, model.feature_importances_):
        importance_rows.append({
            "fold": fold_idx,
            "feature": feature_name,
            "importance": importance_value,
        })

    print(
        f"Fold {fold_idx}: "
        f"train={len(train_idx):,}, test={len(test_idx):,}, "
        f"train_participanți={groups_train.nunique()}, "
        f"test_participanți={groups_test.nunique()}, "
        f"accuracy={acc:.4f}, "
        f"balanced_accuracy={bal_acc:.4f}, "
        f"macro_F1={f1_macro:.4f}, "
        f"weighted_F1={f1_weighted:.4f}"
    )


# ------------------------------------------------------------
# 7. Rezumat rezultate
# ------------------------------------------------------------

results_df = pd.DataFrame(fold_results)

summary_df = pd.DataFrame([{
    "model": "DecisionTreeClassifier",
    "n_splits": N_SPLITS,
    "features": ", ".join(FEATURES),
    "criterion": model.criterion,
    "max_depth": model.max_depth,
    "min_samples_leaf": model.min_samples_leaf,
    "min_samples_split": model.min_samples_split,
    "class_weight": "balanced",
    "ccp_alpha": model.ccp_alpha,
    "tree_depth_final": model.get_depth(),
    "n_leaves_final": model.tree_.n_leaves,
    "accuracy_mean": results_df["accuracy"].mean(),
    "accuracy_std": results_df["accuracy"].std(),
    "balanced_accuracy_mean": results_df["balanced_accuracy"].mean(),
    "balanced_accuracy_std": results_df["balanced_accuracy"].std(),
    "f1_macro_mean": results_df["f1_macro"].mean(),
    "f1_macro_std": results_df["f1_macro"].std(),
    "f1_weighted_mean": results_df["f1_weighted"].mean(),
    "f1_weighted_std": results_df["f1_weighted"].std(),
}])

print_section("Rezumat Decision Tree")
print(summary_df)


# ------------------------------------------------------------
# 8. Matrice de confuzie totală
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
# 9. Importanța variabilelor
# ------------------------------------------------------------

importance_df = pd.DataFrame(importance_rows)

importance_summary = (
    importance_df
    .groupby("feature")["importance"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

print_section("Importanța medie a variabilelor")
print(importance_summary)


# ------------------------------------------------------------
# 10. Comparație cu modelele anterioare
# ------------------------------------------------------------

comparison_rows = []

baseline_path = REPORTS_DIR / "03a_baseline_clasificare_summary.csv"
mnlogit_path = REPORTS_DIR / "03b3_mnlogit_groupkfold_summary.csv"

if baseline_path.exists():
    baseline = pd.read_csv(baseline_path)
    comparison_rows.append({
        "model": "DummyClassifier_most_frequent",
        "accuracy_mean": baseline.loc[0, "accuracy_mean"],
        "balanced_accuracy_mean": baseline.loc[0, "balanced_accuracy_mean"],
        "f1_macro_mean": baseline.loc[0, "f1_macro_mean"],
        "f1_weighted_mean": baseline.loc[0, "f1_weighted_mean"],
    })

if mnlogit_path.exists():
    mnlogit = pd.read_csv(mnlogit_path)
    comparison_rows.append({
        "model": "MNLogit_GroupKFold",
        "accuracy_mean": mnlogit.loc[0, "accuracy_mean"],
        "balanced_accuracy_mean": mnlogit.loc[0, "balanced_accuracy_mean"],
        "f1_macro_mean": mnlogit.loc[0, "f1_macro_mean"],
        "f1_weighted_mean": mnlogit.loc[0, "f1_weighted_mean"],
    })

comparison_rows.append({
    "model": "DecisionTreeClassifier",
    "accuracy_mean": summary_df.loc[0, "accuracy_mean"],
    "balanced_accuracy_mean": summary_df.loc[0, "balanced_accuracy_mean"],
    "f1_macro_mean": summary_df.loc[0, "f1_macro_mean"],
    "f1_weighted_mean": summary_df.loc[0, "f1_weighted_mean"],
})

comparison_df = pd.DataFrame(comparison_rows)

print_section("Comparație modele")
print(comparison_df)

# ------------------------------------------------------------
# 11. Reguli interpretative ale arborelui
# ------------------------------------------------------------

model.fit(X, y)

print("Adâncime arbore:", model.get_depth())
print("Număr frunze:", model.tree_.n_leaves)

tree_rules = export_text(
    model,
    feature_names=FEATURES,
    max_depth=3, 
    show_weights=True
)

rules_path = REPORTS_DIR / "03c_reguli_decision_tree.txt"

with open(rules_path, "w", encoding="utf-8") as f:
    f.write(tree_rules)

print_section("Regulile arborelui de decizie - primele 3 niveluri")
print(tree_rules)

fig, ax = plt.subplots(figsize=(18, 10))

plot_tree(
    model,
    feature_names=FEATURES,
    class_names=["low", "medium", "high"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)

plt.title("Decision Tree final pentru clasificarea nivelului de confidence")
plt.tight_layout()

tree_plot_path = REPORTS_DIR / "03c_decision_tree_final_plot.png"
plt.savefig(tree_plot_path, dpi=300)
plt.close()


# ------------------------------------------------------------
# 12. Salvăm rezultatele
# ------------------------------------------------------------

results_df.to_csv(REPORTS_DIR / "03c_decision_tree_folduri.csv", index=False)
summary_df.to_csv(REPORTS_DIR / "03c_decision_tree_summary.csv", index=False)
cm_df.to_csv(REPORTS_DIR / "03c_decision_tree_confusion_matrix.csv")
importance_summary.to_csv(REPORTS_DIR / "03c_decision_tree_importanta_variabile.csv", index=False)
comparison_df.to_csv(REPORTS_DIR / "03c_comparatie_modele.csv", index=False)
