# 03b2_regresie_logistica_multinomiala_curs.py
# Regresie logistică multinomială - variantă statistică, în stil curs/seminar.
#
# Scop:
# 1. Estimăm un model MNLogit cu statsmodels.
# 2. Obținem coeficienți, erori standard, z-values, p-values.
# 3. Calculăm odds ratios pentru interpretare.
# 4. Verificăm multicolinearitatea prin VIF.
# 5. Salvăm rezultate utile pentru lucrare.
#
# Notă:
# Acest script este pentru interpretare statistică.
# Evaluarea predictivă prin GroupKFold pentru MNLogit este realizată separat în 03b3_mnlogit_groupkfold.py.

from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
CLASS_NAMES = {
    0: "low confidence",
    1: "medium confidence",
    2: "high confidence",
}

MAX_ROWS_STATSMODELS = None

RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. Funcții simple
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    else:
        return ""


def interpret_odds_ratio(or_value):
    if or_value > 1:
        return "crește șansa relativă"
    elif or_value < 1:
        return "scade șansa relativă"
    else:
        return "nu modifică șansa relativă"


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

print_section("Dataset pentru MNLogit")
print("Dataset:", DATA_PATH)
print("Shape după dropna:", df.shape)
print("Număr experimente:", df["experiment_id"].nunique())
print("Număr participanți:", df[GROUP_COL].nunique())

print()
print("Distribuția targetului:")
target_dist = df[TARGET].value_counts().sort_index().reset_index()
target_dist.columns = ["class", "n"]
target_dist["percent"] = target_dist["n"] / target_dist["n"].sum()
target_dist["label"] = target_dist["class"].map(CLASS_NAMES)
print(target_dist)


# ------------------------------------------------------------
# 4. Eșantion opțional pentru statsmodels
# ------------------------------------------------------------

if MAX_ROWS_STATSMODELS is not None and len(df) > MAX_ROWS_STATSMODELS:
    frac = MAX_ROWS_STATSMODELS / len(df)

    data_model = (
        df.groupby(TARGET, group_keys=False)
        .sample(frac=frac, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print_section("Eșantionare pentru statsmodels")
    print("ATENȚIE: modelul statistic rulează pe un eșantion, nu pe toate datele.")
    print("MAX_ROWS_STATSMODELS:", MAX_ROWS_STATSMODELS)
    print("Shape eșantion:", data_model.shape)

else:
    data_model = df.copy()
    print_section("Eșantionare pentru statsmodels")
    print("Modelul statistic rulează pe toate observațiile disponibile.")


# ------------------------------------------------------------
# 5. Pregătire X și y pentru MNLogit
# ------------------------------------------------------------

X = data_model[FEATURES].copy()
y = data_model[TARGET].copy()

# În statsmodels trebuie să adăugăm explicit interceptul.
X_stat = sm.add_constant(X)

print_section("Model specificat")
print("Model:")
print("confidence_class ~ " + " + ".join(FEATURES))
print()
print("Clase:")
print("0 = low confidence")
print("1 = medium confidence")
print("2 = high confidence")
print()
print(
    "În MNLogit, clasa 0 este tratată ca referință. "
    "Coeficienții se interpretează pentru medium vs low și high vs low."
)


# ------------------------------------------------------------
# 6. VIF - verificare multicolinearitate
# ------------------------------------------------------------

print_section("VIF - verificare multicolinearitate")

vif_rows = []

for i, col in enumerate(X_stat.columns):
    if col == "const":
        continue

    vif_value = variance_inflation_factor(X_stat.values, i)

    vif_rows.append({
        "feature": col,
        "VIF": vif_value,
    })

vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

print(vif_df)

vif_df.to_csv(
    REPORTS_DIR / "03b2_mnlogit_vif.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Estimare model MNLogit
# ------------------------------------------------------------

print_section("Estimare model MNLogit")

logit_model = sm.MNLogit(y, X_stat)

try:
    result = logit_model.fit(
        method="bfgs",
        maxiter=500,
        disp=0
    )

    print(result.summary())

    with open(
        REPORTS_DIR / "03b2_mnlogit_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result.summary().as_text())

except Exception as e:
    print("Modelul MNLogit a întâmpinat o eroare:")
    print(e)
    raise


# ------------------------------------------------------------
# 8. Salvăm statistici generale ale modelului
# ------------------------------------------------------------

fit_stats = pd.DataFrame([{
    "model": "MNLogit",
    "n_rows": len(data_model),
    "n_features": len(FEATURES),
    "base_class": "0 = low confidence",
    "log_likelihood": result.llf,
    "ll_null": result.llnull,
    "llr": result.llr,
    "llr_pvalue": result.llr_pvalue,
    "pseudo_r2_mcfadden": result.prsquared,
    "converged": result.mle_retvals.get("converged", np.nan),
}])

print_section("Statistici generale model")
print(fit_stats)

fit_stats.to_csv(
    REPORTS_DIR / "03b2_mnlogit_fit_statistics.csv",
    index=False
)


# ------------------------------------------------------------
# 9. Coeficienți, p-values, odds ratios
# ------------------------------------------------------------

print_section("Coeficienți, p-values și odds ratios")

params = result.params
bse = result.bse
pvalues = result.pvalues

# Pentru clasele 0,1,2, modelul estimează două comparații:
# 1 vs 0 și 2 vs 0.
non_base_classes = [1, 2]

coef_rows = []

for col_position, param_col in enumerate(params.columns):

    compared_class = non_base_classes[col_position]
    comparison_name = f"{CLASS_NAMES[compared_class]} vs {CLASS_NAMES[0]}"

    for feature in params.index:

        coef = params.loc[feature, param_col]
        std_err = bse.loc[feature, param_col]
        pval = pvalues.loc[feature, param_col]

        z_value = coef / std_err

        ci_low = coef - 1.96 * std_err
        ci_high = coef + 1.96 * std_err

        odds_ratio = np.exp(coef)
        odds_ratio_ci_low = np.exp(ci_low)
        odds_ratio_ci_high = np.exp(ci_high)

        coef_rows.append({
            "comparison": comparison_name,
            "compared_class": compared_class,
            "base_class": 0,
            "feature": feature,
            "coef_log_odds": coef,
            "std_error": std_err,
            "z_value": z_value,
            "p_value": pval,
            "significance": significance_stars(pval),
            "odds_ratio": odds_ratio,
            "odds_ratio_ci_low": odds_ratio_ci_low,
            "odds_ratio_ci_high": odds_ratio_ci_high,
            "interpretare_or": interpret_odds_ratio(odds_ratio),
        })

coef_table = pd.DataFrame(coef_rows)

print(coef_table)

coef_table.to_csv(
    REPORTS_DIR / "03b2_mnlogit_coeficienti_odds_ratios.csv",
    index=False
)


# ------------------------------------------------------------
# 10. Variantă ISLP summarize
# ------------------------------------------------------------

print_section("ISLP summarize - opțional")

try:
    from ISLP.models import summarize

    try:
        islp_summary = summarize(result)
        print(islp_summary)

        islp_summary.to_csv(
            REPORTS_DIR / "03b2_mnlogit_islp_summary.csv",
            index=True
        )

        print("ISLP summarize a fost salvat.")
    except Exception as e:
        print("ISLP este instalat, dar summarize nu a funcționat pentru MNLogit.")
        print("Motiv:", e)

except ImportError:
    print("ISLP nu este instalat. Nu este obligatoriu pentru acest script.")
    print("Dacă vrei să îl instalezi: pip install ISLP")


# ------------------------------------------------------------
# 11. Predicții probabilistice in-sample
# ------------------------------------------------------------

print_section("Predicții probabilistice in-sample")

pred_probs_raw = result.predict(X_stat)

# statsmodels poate întoarce un DataFrame cu coloane 0, 1, 2.
# Convertim explicit la array pentru a evita apariția valorilor NaN.
pred_probs_array = np.asarray(pred_probs_raw)

pred_probs_df = pd.DataFrame(
    pred_probs_array,
    columns=[
        "prob_low_0",
        "prob_medium_1",
        "prob_high_2",
    ]
)

pred_class = pred_probs_array.argmax(axis=1)

in_sample_metrics = pd.DataFrame([{
    "model": "MNLogit_in_sample",
    "accuracy": accuracy_score(y, pred_class),
    "balanced_accuracy": balanced_accuracy_score(y, pred_class),
    "f1_macro": f1_score(y, pred_class, average="macro"),
    "f1_weighted": f1_score(y, pred_class, average="weighted"),
}])

print("Primele 10 predicții probabilistice:")
print(pred_probs_df.head(10))

print()
print("Metrici in-sample:")
print(in_sample_metrics)

in_sample_metrics.to_csv(
    REPORTS_DIR / "03b2_mnlogit_in_sample_metrics.csv",
    index=False
)

pred_probs_df.head(1000).to_csv(
    REPORTS_DIR / "03b2_mnlogit_predicted_probabilities_sample.csv",
    index=False
)


# ------------------------------------------------------------
# 12. Matrice de confuzie in-sample
# ------------------------------------------------------------

cm = confusion_matrix(y, pred_class, labels=CLASS_LABELS)

cm_df = pd.DataFrame(
    cm,
    index=["true_low_0", "true_medium_1", "true_high_2"],
    columns=["pred_low_0", "pred_medium_1", "pred_high_2"]
)

print_section("Matrice de confuzie in-sample")
print(cm_df)

cm_df.to_csv(
    REPORTS_DIR / "03b2_mnlogit_in_sample_confusion_matrix.csv"
)


# ------------------------------------------------------------
# 13. Classification report in-sample
# ------------------------------------------------------------

report_dict = classification_report(
    y,
    pred_class,
    labels=CLASS_LABELS,
    target_names=["low_0", "medium_1", "high_2"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).T

print_section("Classification report in-sample")
print(report_df)

report_df.to_csv(
    REPORTS_DIR / "03b2_mnlogit_in_sample_classification_report.csv"
)

