# 01c_analiza_target_predictori.py
# Analiza relațiilor dintre predictorii candidați și confidence_class.
#
# Scop:
# 1. Construim un dataset exploratoriu pentru clasificare.
# 2. Recodificăm Confidence în 3 clase:
#    0 = low confidence
#    1 = medium confidence
#    2 = high confidence
# 3. Analizăm relațiile dintre confidence_class și predictorii candidați:
#    - Accuracy
#    - RT_dec
#    - Stimulus / Response
#    - RT_conf
#    - Condition, Difficulty, Contrast, Coherence
#    - Trial, Block, Task, Training
#    - Age, Gender
# 4. Salvăm statistici descriptive, corelații și figuri.

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = PROJECT_DIR / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("Folder proiect:", PROJECT_DIR)
print("Folder date raw:", RAW_DIR)
print("Folder rapoarte:", REPORTS_DIR)
print("Folder figuri:", FIGURES_DIR)


# ------------------------------------------------------------
# 2. Funcții simple ajutătoare
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def find_column_case_insensitive(columns, possible_names):
    lower_to_original = {str(col).lower(): col for col in columns}

    for name in possible_names:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    return None


def normalize_minmax(series):
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.nan, index=s.index)

    return (s - min_val) / (max_val - min_val)


def recodeaza_confidence_3clase(series):
    """
    0 = low confidence
    1 = medium confidence
    2 = high confidence
    """
    conf_01 = normalize_minmax(series)

    confidence_class = pd.cut(
        conf_01,
        bins=[-0.001, 1/3, 2/3, 1.001],
        labels=[0, 1, 2],
        include_lowest=True
    )

    return confidence_class.astype("float")


def zscore_safe(series):
    s = pd.to_numeric(series, errors="coerce")
    mean_val = s.mean()
    std_val = s.std(ddof=0)

    if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
        return pd.Series(np.nan, index=s.index)

    return (s - mean_val) / std_val


def cramers_v(x, y):
    """
    Măsoară asocierea dintre două variabile categorice.
    Valori apropiate de 0 = asociere slabă.
    Valori mai mari = asociere mai puternică.
    """
    table = pd.crosstab(x, y)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()

    phi2 = chi2 / n
    r, k = table.shape

    return np.sqrt(phi2 / min(k - 1, r - 1))


# ------------------------------------------------------------
# 3. Coloane posibile
# ------------------------------------------------------------

COLUMN_GROUPS = {
    "Confidence": ["Confidence", "confidence"],
    "RT_dec": ["RT_dec", "rt_dec", "rt", "RT"],
    "Accuracy": ["Accuracy", "accuracy", "correct", "Correct"],
    "Subj_idx": ["Subj_idx", "subj_idx"],
    "Stimulus": ["Stimulus", "stimulus"],
    "Response": ["Response", "response"],
    "RT_conf": ["RT_conf", "rt_conf"],
    "Condition": ["Condition", "condition"],
    "Difficulty": ["Difficulty", "difficulty"],
    "Contrast": ["Contrast", "contrast"],
    "Coherence": ["Coherence", "coherence"],
    "Trial": ["Trial", "trial"],
    "Block": ["Block", "block"],
    "Task": ["Task", "task"],
    "Training": ["Training"],
    "Age": ["Age", "age"],
    "Gender": ["Gender"],
}


NUMERIC_PREDICTORS = [
    "rt_dec_clean_log_z",
    "accuracy",
    "rt_x_acc",
    "rt_conf_log_z",
    "Difficulty",
    "Contrast",
    "Coherence",
    "Trial",
    "Block",
    "Age",
]

CATEGORICAL_PREDICTORS = [
    "Stimulus",
    "Response",
    "Condition",
    "Task",
    "Training",
    "Gender",
]


# ------------------------------------------------------------
# 4. Regula conservatoare pentru RT_dec
# ------------------------------------------------------------

RT_MIN_SECONDS = 0.1
RT_MAX_SECONDS = 10

EXCLUDE_EXPERIMENT_IF_MEDIAN_ABOVE = 10
EXCLUDE_EXPERIMENT_IF_PERCENT_ABOVE_10 = 0.50


# ------------------------------------------------------------
# 5. Încărcăm lista fișierelor incluse
# ------------------------------------------------------------

target_summary_path = REPORTS_DIR / "01_target_confidence_summary_cu_includere.csv"

if not target_summary_path.exists():
    raise FileNotFoundError(
        "Nu găsesc reports/01_target_confidence_summary_cu_includere.csv. "
        "Rulează mai întâi 01_explorare_date.py."
    )

target_summary = pd.read_csv(target_summary_path)

included_files = target_summary.loc[
    target_summary["inclusion_status"] == "include_scala_interpretabile",
    "file"
].tolist()

print_section("Fișiere disponibile după filtrarea targetului")
print("Număr fișiere incluse:", len(included_files))


# ------------------------------------------------------------
# 6. Construim datasetul exploratoriu
# ------------------------------------------------------------
# Cerem obligatoriu:
# - Confidence
# - RT_dec
# - Accuracy
# - Subj_idx
#
# Restul predictorilor sunt opționali și se adaugă dacă există.

parts = []
log_rows = []

for file_name in included_files:
    file = RAW_DIR / file_name

    try:
        columns = pd.read_csv(file, nrows=0).columns.tolist()

        conf_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Confidence"])
        rt_col = find_column_case_insensitive(columns, COLUMN_GROUPS["RT_dec"])
        acc_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Accuracy"])
        subj_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Subj_idx"])

        if conf_col is None or rt_col is None or acc_col is None or subj_col is None:
            log_rows.append({
                "file": file_name,
                "status": "skip_missing_core_columns",
                "rows_after": 0
            })
            continue

        usecols = [conf_col, rt_col, acc_col, subj_col]

        optional_cols = {}

        for var in [
            "Stimulus", "Response", "RT_conf", "Condition", "Difficulty",
            "Contrast", "Coherence", "Trial", "Block", "Task",
            "Training", "Age", "Gender"
        ]:
            col = find_column_case_insensitive(columns, COLUMN_GROUPS[var])
            if col is not None:
                usecols.append(col)
                optional_cols[col] = var

        df = pd.read_csv(file, usecols=usecols)

        rename_map = {
            conf_col: "confidence",
            rt_col: "rt_dec",
            acc_col: "accuracy",
            subj_col: "subj_id",
        }

        for original_col, new_col in optional_cols.items():
            rename_map[original_col] = new_col

        df = df.rename(columns=rename_map)

        experiment_id = Path(file_name).stem

        # Conversii
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["rt_dec"] = pd.to_numeric(df["rt_dec"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

        df = df.dropna(subset=["confidence", "rt_dec", "accuracy", "subj_id"])
        df = df[df["rt_dec"] > 0]
        df = df[df["accuracy"].isin([0, 1])]

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_valid_rows",
                "rows_after": 0
            })
            continue

        # Normalizare și recodificare target
        df["confidence_01"] = normalize_minmax(df["confidence"])
        df["confidence_class"] = recodeaza_confidence_3clase(df["confidence"])

        df = df.dropna(subset=["confidence_01", "confidence_class"])

        # Diagnostic RT_dec la nivel de experiment
        median_rt = df["rt_dec"].median()
        percent_rt_above_10 = (df["rt_dec"] > 10).mean()

        exclude_experiment = (
            (median_rt > EXCLUDE_EXPERIMENT_IF_MEDIAN_ABOVE)
            or (percent_rt_above_10 > EXCLUDE_EXPERIMENT_IF_PERCENT_ABOVE_10)
        )

        if exclude_experiment:
            log_rows.append({
                "file": file_name,
                "status": "exclude_rt_neplauzibil",
                "rows_after": 0,
                "median_rt": median_rt,
                "percent_rt_above_10": percent_rt_above_10
            })
            continue

        # Curățare RT_dec la nivel de observație
        df = df[
            (df["rt_dec"] >= RT_MIN_SECONDS)
            & (df["rt_dec"] <= RT_MAX_SECONDS)
        ].copy()

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_rows_after_rt_filter",
                "rows_after": 0
            })
            continue

        # Transformări RT_dec
        df["rt_dec_clean"] = df["rt_dec"]
        df["rt_dec_log"] = np.log(df["rt_dec_clean"])
        df["rt_dec_clean_log_z"] = zscore_safe(df["rt_dec_log"])

        # Interacțiune RT x Accuracy
        df["rt_x_acc"] = df["rt_dec_clean_log_z"] * df["accuracy"]

        # Dacă există RT_conf, îl transformăm similar
        if "RT_conf" in df.columns:
            df["RT_conf"] = pd.to_numeric(df["RT_conf"], errors="coerce")
            df.loc[df["RT_conf"] <= 0, "RT_conf"] = np.nan
            df["rt_conf_log"] = np.log(df["RT_conf"])
            df["rt_conf_log_z"] = zscore_safe(df["rt_conf_log"])

        # Convertim unele variabile numerice dacă există
        for col in ["Difficulty", "Contrast", "Coherence", "Trial", "Block", "Age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["experiment_id"] = experiment_id
        df["subj_id"] = df["subj_id"].astype(str)
        df["participant_uid"] = df["experiment_id"] + "_" + df["subj_id"]

        keep_cols = [
            "experiment_id",
            "subj_id",
            "participant_uid",
            "confidence_01",
            "confidence_class",
            "accuracy",
            "rt_dec_clean",
            "rt_dec_clean_log_z",
            "rt_x_acc",
            "Stimulus",
            "Response",
            "RT_conf",
            "rt_conf_log_z",
            "Condition",
            "Difficulty",
            "Contrast",
            "Coherence",
            "Trial",
            "Block",
            "Task",
            "Training",
            "Age",
            "Gender",
        ]

        keep_cols_existing = [col for col in keep_cols if col in df.columns]

        df = df[keep_cols_existing]
        df = df.replace([np.inf, -np.inf], np.nan)

        parts.append(df)

        log_rows.append({
            "file": file_name,
            "status": "included",
            "rows_after": len(df),
            "median_rt": median_rt,
            "percent_rt_above_10": percent_rt_above_10
        })

    except Exception as e:
        log_rows.append({
            "file": file_name,
            "status": "error",
            "error": str(e),
            "rows_after": 0
        })


df_all = pd.concat(parts, ignore_index=True)
log_df = pd.DataFrame(log_rows)

log_df.to_csv(REPORTS_DIR / "01c_log_constructie_dataset_relatii.csv", index=False)

print_section("Dataset exploratoriu pentru relații")
print("Shape:", df_all.shape)
print("Număr experimente:", df_all["experiment_id"].nunique())
print("Număr participanți:", df_all["participant_uid"].nunique())

print()
print("Status fișiere:")
print(log_df["status"].value_counts())


# ------------------------------------------------------------
# 7. Distribuția targetului
# ------------------------------------------------------------

print_section("Distribuția confidence_class")

class_counts = (
    df_all["confidence_class"]
    .value_counts()
    .sort_index()
    .reset_index()
)

class_counts.columns = ["confidence_class", "n_rows"]
class_counts["percent"] = class_counts["n_rows"] / class_counts["n_rows"].sum()

print(class_counts)

class_counts.to_csv(
    REPORTS_DIR / "01c_distributie_confidence_class.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Statistici descriptive generale
# ------------------------------------------------------------

print_section("Statistici descriptive pentru variabile numerice")

numeric_cols_existing = [
    col for col in NUMERIC_PREDICTORS + ["confidence_01", "confidence_class"]
    if col in df_all.columns
]

desc_numeric = df_all[numeric_cols_existing].describe().T

print(desc_numeric)

desc_numeric.to_csv(
    REPORTS_DIR / "01c_statistici_descriptive_numerice.csv",
    index=True
)


# ------------------------------------------------------------
# 9. Relația Accuracy - confidence_class
# ------------------------------------------------------------

print_section("Relația dintre Accuracy și confidence_class")

acc_table = pd.crosstab(
    df_all["confidence_class"],
    df_all["accuracy"],
    normalize="index"
)

print("Distribuție Accuracy în interiorul fiecărei clase de confidence:")
print(acc_table)

acc_summary = (
    df_all
    .groupby("confidence_class")["accuracy"]
    .agg(["count", "mean", "std"])
    .reset_index()
)

print()
print("Media Accuracy pe clase de confidence:")
print(acc_summary)

acc_summary.to_csv(
    REPORTS_DIR / "01c_accuracy_pe_clase_confidence.csv",
    index=False
)


# ------------------------------------------------------------
# 10. Relația variabile numerice - confidence_class
# ------------------------------------------------------------

print_section("Relația predictorilor numerici cu confidence_class")

numeric_relation_rows = []

for predictor in NUMERIC_PREDICTORS:
    if predictor not in df_all.columns:
        continue

    temp = df_all[[predictor, "confidence_class"]].dropna()

    if len(temp) < 100:
        continue

    rho, p_value = spearmanr(temp[predictor], temp["confidence_class"])

    grouped = (
        temp
        .groupby("confidence_class")[predictor]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    for _, row in grouped.iterrows():
        numeric_relation_rows.append({
            "predictor": predictor,
            "confidence_class": row["confidence_class"],
            "count": row["count"],
            "mean": row["mean"],
            "std": row["std"],
            "median": row["median"],
            "spearman_rho_global": rho,
        })

numeric_relations = pd.DataFrame(numeric_relation_rows)

print(numeric_relations.head(50))

numeric_relations.to_csv(
    REPORTS_DIR / "01c_relatii_predictori_numerici_confidence_class.csv",
    index=False
)

# Rezumat pe predictor
numeric_strength = (
    numeric_relations
    .groupby("predictor")["spearman_rho_global"]
    .first()
    .reset_index()
)

numeric_strength["abs_rho"] = numeric_strength["spearman_rho_global"].abs()
numeric_strength = numeric_strength.sort_values("abs_rho", ascending=False)

print()
print("Forța asocierii numerice cu confidence_class:")
print(numeric_strength)

numeric_strength.to_csv(
    REPORTS_DIR / "01c_corelatii_numerice_cu_confidence_class.csv",
    index=False
)


# ------------------------------------------------------------
# 11. Relația variabile categorice - confidence_class
# ------------------------------------------------------------

print_section("Relația predictorilor categorici cu confidence_class")

categorical_rows = []

for predictor in CATEGORICAL_PREDICTORS:
    if predictor not in df_all.columns:
        continue

    temp = df_all[[predictor, "confidence_class"]].dropna()

    if len(temp) < 100:
        continue

    n_categories = temp[predictor].nunique()

    # Evităm variabile cu prea multe categorii distincte.
    # Dacă are foarte multe valori, devine greu de interpretat ca predictor categoric simplu.
    if n_categories > 30:
        note = "prea_multe_categorii"
        cv = np.nan
    else:
        note = "ok"
        cv = cramers_v(temp[predictor], temp["confidence_class"])

    categorical_rows.append({
        "predictor": predictor,
        "n_rows": len(temp),
        "n_categories": n_categories,
        "cramers_v": cv,
        "note": note,
    })

categorical_summary = (
    pd.DataFrame(categorical_rows)
    .sort_values("cramers_v", ascending=False)
)

print(categorical_summary)

categorical_summary.to_csv(
    REPORTS_DIR / "01c_asocieri_categorice_cu_confidence_class.csv",
    index=False
)


# ------------------------------------------------------------
# 12. Matrice de corelație pentru variabile numerice
# ------------------------------------------------------------
# Pentru că datasetul este foarte mare, folosim un eșantion pentru matrice și figuri.
# Corelațiile sunt exploratorii.

print_section("Matrice de corelație")

corr_cols = [
    "confidence_class",
    "confidence_01",
    "accuracy",
    "rt_dec_clean_log_z",
    "rt_x_acc",
    "rt_conf_log_z",
    "Difficulty",
    "Contrast",
    "Coherence",
    "Trial",
    "Block",
    "Age",
]

corr_cols_existing = [col for col in corr_cols if col in df_all.columns]

sample_size = min(200000, len(df_all))
df_sample = df_all.sample(n=sample_size, random_state=42)

corr_matrix = df_sample[corr_cols_existing].corr(method="spearman")

print(corr_matrix)

corr_matrix.to_csv(
    REPORTS_DIR / "01c_matrice_corelatie_spearman.csv",
    index=True
)


# ------------------------------------------------------------
# 13. Figuri reprezentative
# ------------------------------------------------------------

# 13.1 Distribuția claselor
plt.figure(figsize=(6, 4))
plt.bar(class_counts["confidence_class"].astype(str), class_counts["n_rows"])
plt.title("Distribuția confidence_class")
plt.xlabel("confidence_class")
plt.ylabel("Număr observații")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01c_distributie_confidence_class.png", dpi=150)
plt.close()

# 13.2 Accuracy medie pe clase de confidence
plt.figure(figsize=(6, 4))
plt.bar(acc_summary["confidence_class"].astype(str), acc_summary["mean"])
plt.title("Accuracy medie pe clase de confidence")
plt.xlabel("confidence_class")
plt.ylabel("Accuracy medie")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01c_accuracy_medie_pe_confidence_class.png", dpi=150)
plt.close()

# 13.3 RT_dec pe clase de confidence
plot_df = df_sample.dropna(subset=["rt_dec_clean_log_z", "confidence_class"])

data_to_plot = [
    plot_df.loc[plot_df["confidence_class"] == cls, "rt_dec_clean_log_z"]
    for cls in sorted(plot_df["confidence_class"].dropna().unique())
]

plt.figure(figsize=(7, 4))
plt.boxplot(data_to_plot, labels=[str(int(cls)) for cls in sorted(plot_df["confidence_class"].dropna().unique())])
plt.title("RT_dec log z-score pe clase de confidence")
plt.xlabel("confidence_class")
plt.ylabel("RT_dec log z-score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01c_boxplot_rt_dec_pe_confidence_class.png", dpi=150)
plt.close()

# 13.4 Forța asocierii numerice
plt.figure(figsize=(8, 5))
plt.barh(numeric_strength["predictor"], numeric_strength["abs_rho"])
plt.gca().invert_yaxis()
plt.title("Forța asocierii predictorilor numerici cu confidence_class")
plt.xlabel("|Spearman rho|")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01c_asocieri_numerice_confidence_class.png", dpi=150)
plt.close()

# 13.5 Asocieri categorice
if len(categorical_summary) > 0:
    cat_plot = categorical_summary.dropna(subset=["cramers_v"])

    plt.figure(figsize=(8, 5))
    plt.barh(cat_plot["predictor"], cat_plot["cramers_v"])
    plt.gca().invert_yaxis()
    plt.title("Asocierea predictorilor categorici cu confidence_class")
    plt.xlabel("Cramer's V")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01c_asocieri_categorice_confidence_class.png", dpi=150)
    plt.close()

# 13.6 Matrice corelație
plt.figure(figsize=(9, 7))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Spearman rho")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Matrice de corelație Spearman")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01c_matrice_corelatie_spearman.png", dpi=150)
plt.close()
