# 04_roc_auc_modele.py
# Curbe ROC/AUC multiclasă pentru modelele finale.
#
# Modele incluse:
# 1. MNLogit statsmodels, evaluat out-of-sample prin GroupKFold
# 2. Decision Tree
# 3. KNN
# 4. LinearSVC
#
# Nu includem LogisticRegression sklearn.
# Nu includem MNLogit in-sample.
#
# Pentru target multiclasă folosim abordarea One-vs-Rest:
# - low vs rest
# - medium vs rest
# - high vs rest
#
# Scorurile sunt obținute out-of-fold prin GroupKFold,
# pentru a evita evaluarea pe aceleași date pe care modelul a fost antrenat.

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = PROJECT_DIR / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DATA_PATH = PROCESSED_DIR / "dataset_main_classification.csv"

TARGET = "confidence_class"
GROUP_COL = "uid_participant"

N_SPLITS = 3
RANDOM_STATE = 42

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}


# ------------------------------------------------------------
# 2. Funcții ajutătoare
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def safe_file_name(model_name):
    return (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def get_sklearn_models():
    """
    Modelele sklearn păstrate în comparația finală.
    LogisticRegression sklearn nu mai este inclus.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight="balanced",
            ccp_alpha=0.0,
            random_state=RANDOM_STATE
        ),

        "KNN": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=21,
                weights="uniform",
                metric="manhattan",
                n_jobs=-1
            ))
        ]),

        "LinearSVC": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=1,
                class_weight="balanced",
                dual=False,
                max_iter=20000,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    return models


def get_model_scores(model, X_test):
    """
    ROC poate fi calculat pe probabilități sau pe scoruri continue.

    Decision Tree și KNN folosesc predict_proba.
    LinearSVC folosește decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise ValueError("Modelul nu are predict_proba sau decision_function.")


def compute_oof_scores_sklearn(model, X, y, groups):
    """
    Returnează scoruri out-of-fold pentru modelele sklearn.
    Fiecare observație primește scoruri de la un model care nu a fost
    antrenat pe participantul respectiv.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_scores = np.zeros((len(X), len(CLASS_LABELS)))

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)

        scores = get_model_scores(model, X_test)

        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        if scores.shape[1] != len(CLASS_LABELS):
            raise ValueError(
                f"Modelul sklearn a returnat {scores.shape[1]} coloane, "
                f"dar erau așteptate {len(CLASS_LABELS)}."
            )

        oof_scores[test_idx, :] = scores

        print(
            f"Fold {fold_idx}: train={len(train_idx):,}, "
            f"test={len(test_idx):,}, scores shape={scores.shape}"
        )

    return oof_scores


def compute_oof_scores_mnlogit(X, y, groups):
    """
    Probabilități out-of-fold pentru MNLogit statsmodels.

    Fiecare observație primește probabilități estimate de un model antrenat
    fără participantul respectiv.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_scores = np.zeros((len(X), len(CLASS_LABELS)))

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()

        X_train_const = sm.add_constant(X_train, has_constant="add")
        X_test_const = sm.add_constant(X_test, has_constant="add")

        model = sm.MNLogit(y_train, X_train_const)

        result = model.fit(
            method="lbfgs",
            maxiter=200,
            disp=False
        )

        probs = result.predict(X_test_const)

        if isinstance(probs, pd.DataFrame):
            probs = probs.reindex(columns=CLASS_LABELS)
            probs = probs.to_numpy()
        else:
            probs = np.asarray(probs)

        if probs.shape[1] != len(CLASS_LABELS):
            raise ValueError(
                f"MNLogit a returnat {probs.shape[1]} coloane, "
                f"dar erau așteptate {len(CLASS_LABELS)}."
            )

        oof_scores[test_idx, :] = probs

        print(
            f"Fold {fold_idx}: train={len(train_idx):,}, "
            f"test={len(test_idx):,}, scores shape={probs.shape}, "
            f"converged={result.mle_retvals.get('converged', None)}"
        )

    return oof_scores


def calculate_roc_auc(model_name, y_true, y_scores):
    """
    Calculează AUC pentru fiecare clasă, macro AUC, weighted AUC și micro AUC.
    """
    y_bin = label_binarize(y_true, classes=CLASS_LABELS)

    rows = []
    roc_data = {}

    supports = y_bin.sum(axis=0)

    for class_idx, class_label in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_scores[:, class_idx])
        class_auc = auc(fpr, tpr)

        roc_data[class_label] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": class_auc,
        }

        rows.append({
            "model": model_name,
            "class": class_label,
            "class_name": CLASS_NAMES[class_label],
            "auc_ovr": class_auc,
            "support": int(supports[class_idx]),
        })

    class_auc_values = np.array([row["auc_ovr"] for row in rows])
    support_values = np.array([row["support"] for row in rows])

    macro_auc = class_auc_values.mean()
    weighted_auc = np.average(class_auc_values, weights=support_values)

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_scores.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)

    summary_row = {
        "model": model_name,
        "auc_macro_ovr": macro_auc,
        "auc_weighted_ovr": weighted_auc,
        "auc_micro_ovr": micro_auc,
    }

    return pd.DataFrame(rows), summary_row, roc_data


def plot_model_roc(model_name, roc_data):
    """
    Salvează câte un grafic ROC One-vs-Rest pentru fiecare model.
    """
    plt.figure(figsize=(8, 6))

    for class_label in CLASS_LABELS:
        data = roc_data[class_label]

        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{CLASS_NAMES[class_label]} vs rest, AUC={data['auc']:.3f}"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Clasificator aleatoriu")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title(f"Curbe ROC One-vs-Rest - {model_name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"04_roc_auc_{safe_file_name(model_name)}.png",
        dpi=300
    )

    plt.close()


def plot_macro_auc_comparison(summary_df):
    """
    Grafic comparativ cu macro AUC pentru toate modelele finale.
    """
    plot_df = summary_df.sort_values("auc_macro_ovr", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["model"], plot_df["auc_macro_ovr"])

    plt.xlabel("Macro AUC One-vs-Rest")
    plt.title("Comparație modele după Macro AUC")
    plt.xlim(0.45, 0.75)

    for idx, value in enumerate(plot_df["auc_macro_ovr"]):
        plt.text(value + 0.005, idx, f"{value:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_roc_auc_macro_comparison.png", dpi=300)
    plt.close()


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

print_section("ROC/AUC pentru modelele finale")
print("Dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())
print("Predictori:", FEATURES)


# ------------------------------------------------------------
# 4. Calcul ROC/AUC
# ------------------------------------------------------------

all_class_auc = []
all_summary_rows = []


# ------------------------------------------------------------
# 4.1 MNLogit statsmodels out-of-sample
# ------------------------------------------------------------

print_section("Model: MNLogit_GroupKFold")

mnlogit_scores = compute_oof_scores_mnlogit(X, y, groups)

class_auc_df, summary_row, roc_data = calculate_roc_auc(
    "MNLogit_GroupKFold",
    y,
    mnlogit_scores
)

all_class_auc.append(class_auc_df)
all_summary_rows.append(summary_row)

print()
print("AUC pe clase:")
print(class_auc_df)

print()
print("AUC sumar:")
print(summary_row)

plot_model_roc("MNLogit_GroupKFold", roc_data)

class_auc_df.to_csv(
    REPORTS_DIR / "04_roc_auc_mnlogit_groupkfold_class_auc.csv",
    index=False
)


# ------------------------------------------------------------
# 4.2 Modele sklearn finale
# ------------------------------------------------------------

models = get_sklearn_models()

for model_name, model in models.items():
    print_section(f"Model: {model_name}")

    oof_scores = compute_oof_scores_sklearn(model, X, y, groups)

    class_auc_df, summary_row, roc_data = calculate_roc_auc(
        model_name,
        y,
        oof_scores
    )

    all_class_auc.append(class_auc_df)
    all_summary_rows.append(summary_row)

    print()
    print("AUC pe clase:")
    print(class_auc_df)

    print()
    print("AUC sumar:")
    print(summary_row)

    plot_model_roc(model_name, roc_data)

    class_auc_df.to_csv(
        REPORTS_DIR / f"04_roc_auc_{safe_file_name(model_name)}_class_auc.csv",
        index=False
    )


# ------------------------------------------------------------
# 5. Salvăm comparația finală
# ------------------------------------------------------------

class_auc_all_df = pd.concat(all_class_auc, ignore_index=True)

summary_df = pd.DataFrame(all_summary_rows)

summary_df = summary_df.sort_values(
    "auc_macro_ovr",
    ascending=False
).reset_index(drop=True)

print_section("Comparație finală ROC/AUC")
print(summary_df)

class_auc_all_df.to_csv(
    REPORTS_DIR / "04_roc_auc_all_models_class_auc.csv",
    index=False
)

summary_df.to_csv(
    REPORTS_DIR / "04_roc_auc_all_models_summary.csv",
    index=False
)

plot_macro_auc_comparison(summary_df)


print_section("Fișiere salvate")
print(REPORTS_DIR / "04_roc_auc_all_models_class_auc.csv")
print(REPORTS_DIR / "04_roc_auc_all_models_summary.csv")
print(FIGURES_DIR / "04_roc_auc_macro_comparison.png")
print("Grafice ROC separate au fost salvate în folderul figures/.")