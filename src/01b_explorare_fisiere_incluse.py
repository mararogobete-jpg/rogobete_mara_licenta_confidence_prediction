# 01b_explorare_fisiere_incluse.py
# Explorare a fișierelor incluse după analiza targetului.
#
# 1. Folosim doar fișierele cu Confidence interpretabil.
# 2. Normalizăm Confidence în [0,1].
# 3. Recodificăm Confidence în 3 clase ordinale:
#    0 = low confidence
#    1 = medium confidence
#    2 = high confidence
# 4. Verificăm distribuția claselor.
# 5. Verificăm disponibilitatea predictorilor pentru modelare.

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 2. Funcții ajutătoare simple
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
    conf_01 = normalize_minmax(series)

    confidence_class = pd.cut(
        conf_01,
        bins=[-0.001, 1/3, 2/3, 1.001],
        labels=[0, 1, 2],
        include_lowest=True
    )

    return confidence_class.astype("float")


# ------------------------------------------------------------
# 3. Coloane posibile
# ------------------------------------------------------------

COLUMN_GROUPS = {
    "Confidence": ["Confidence", "confidence"],
    "RT_dec": ["RT_dec", "rt_dec", "rt", "RT"],
    "Accuracy": ["Accuracy", "accuracy", "correct", "Correct"],
    "RT_conf": ["RT_conf", "rt_conf"],
    "RT_decConf": ["RT_decConf"],
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
    "Stimulus": ["Stimulus", "stimulus"],
    "Response": ["Response", "response"],
    "Subj_idx": ["Subj_idx", "subj_idx"],
}


# ------------------------------------------------------------
# 4. Încărcăm fișierele incluse din 01
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

print_section("Fișiere incluse")
print("Număr fișiere incluse:", len(included_files))


# ------------------------------------------------------------
# 5. Analizăm Confidence recodificat în 3 clase
# ------------------------------------------------------------

confidence_rows = []
class_rows = []

for file_name in included_files:
    file = RAW_DIR / file_name

    columns = pd.read_csv(file, nrows=0).columns.tolist()
    conf_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Confidence"])

    if conf_col is None:
        continue

    df = pd.read_csv(file, usecols=[conf_col])

    conf = pd.to_numeric(df[conf_col], errors="coerce")
    conf_01 = normalize_minmax(conf)
    conf_class = recodeaza_confidence_3clase(conf)

    counts = conf_class.value_counts(dropna=True).sort_index()

    confidence_rows.append({
        "file": file_name,
        "n_rows": len(conf),
        "n_valid_confidence": conf.notna().sum(),
        "raw_min": conf.min(),
        "raw_max": conf.max(),
        "raw_mean": conf.mean(),
        "raw_std": conf.std(),
        "n_unique_raw": conf.nunique(),
        "confidence_01_mean": conf_01.mean(),
        "confidence_01_std": conf_01.std(),
        "low_0": int(counts.get(0, 0)),
        "medium_1": int(counts.get(1, 0)),
        "high_2": int(counts.get(2, 0)),
    })

    class_rows.append({
        "file": file_name,
        "low_0": int(counts.get(0, 0)),
        "medium_1": int(counts.get(1, 0)),
        "high_2": int(counts.get(2, 0)),
        "total_valid_class": int(conf_class.notna().sum()),
    })


confidence_summary = pd.DataFrame(confidence_rows)
class_summary = pd.DataFrame(class_rows)

confidence_summary.to_csv(
    REPORTS_DIR / "01b_confidence_3clase_per_fisier.csv",
    index=False
)

class_summary.to_csv(
    REPORTS_DIR / "01b_distributie_confidence_3clase_per_fisier.csv",
    index=False
)


print_section("Rezumat Confidence recodificat în 3 clase")

print("Statistici confidence_01 la nivel de experimente:")
print(
    confidence_summary[
        ["confidence_01_mean", "confidence_01_std", "n_unique_raw"]
    ].describe().T
)

total_low = class_summary["low_0"].sum()
total_medium = class_summary["medium_1"].sum()
total_high = class_summary["high_2"].sum()
total_all = total_low + total_medium + total_high

class_global = pd.DataFrame({
    "confidence_class": ["low_0", "medium_1", "high_2"],
    "n_rows": [total_low, total_medium, total_high],
})

class_global["percent"] = class_global["n_rows"] / total_all

class_global.to_csv(
    REPORTS_DIR / "01b_distributie_confidence_3clase_global.csv",
    index=False
)

print()
print("Distribuția globală a claselor:")
print(class_global)


# ------------------------------------------------------------
# 6. Figură: distribuția claselor
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.bar(class_global["confidence_class"], class_global["n_rows"])
plt.title("Distribuția Confidence recodificat în 3 clase")
plt.ylabel("Număr observații")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01b_distributie_confidence_3clase.png", dpi=150)
plt.close()

print()
print("Figură salvată:")
print(" - figures/01b_distributie_confidence_3clase.png")


# ------------------------------------------------------------
# 7. Disponibilitatea predictorilor în fișierele incluse
# ------------------------------------------------------------

predictor_rows = []

for file_name in included_files:
    file = RAW_DIR / file_name
    columns = pd.read_csv(file, nrows=0).columns.tolist()

    row = {"file": file_name}

    for predictor, possible_names in COLUMN_GROUPS.items():
        col = find_column_case_insensitive(columns, possible_names)
        row[predictor] = col is not None

    predictor_rows.append(row)


predictor_df = pd.DataFrame(predictor_rows)

predictor_availability_rows = []

for predictor in COLUMN_GROUPS.keys():
    if predictor == "Confidence":
        continue

    n_files = predictor_df[predictor].sum()
    predictor_availability_rows.append({
        "predictor": predictor,
        "n_files": int(n_files),
        "percent_files": n_files / len(included_files),
    })

predictor_availability = (
    pd.DataFrame(predictor_availability_rows)
    .sort_values("n_files", ascending=False)
    .reset_index(drop=True)
)

predictor_availability.to_csv(
    REPORTS_DIR / "01b_predictori_disponibilitate_summary.csv",
    index=False
)

predictor_df.to_csv(
    REPORTS_DIR / "01b_predictori_disponibilitate_fisiere_incluse.csv",
    index=False
)

print_section("Disponibilitatea predictorilor în fișierele incluse")
print(predictor_availability)


# ------------------------------------------------------------
# 8. Figură: disponibilitatea predictorilor
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(
    predictor_availability["predictor"],
    predictor_availability["n_files"]
)
plt.gca().invert_yaxis()
plt.xlabel("Număr fișiere")
plt.title("Disponibilitatea predictorilor în fișierele incluse")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01b_disponibilitate_predictori.png", dpi=150)
plt.close()

print()
print("Figură salvată:")
print(" - figures/01b_disponibilitate_predictori.png")


# ------------------------------------------------------------
# 9. Simulare seturi de predictori pentru clasificare
# ------------------------------------------------------------

feature_sets = {
    # --------------------------------------------------------
    # Seturi minime
    # --------------------------------------------------------
    "doar_confidence_class": [],

    "classification_RT_dec": [
        "RT_dec"
    ],

    "classification_accuracy": [
        "Accuracy"
    ],

    # --------------------------------------------------------
    # Set principal propus
    # --------------------------------------------------------
    "classification_RT_dec_accuracy": [
        "RT_dec",
        "Accuracy"
    ],

    "classification_RT_dec_accuracy_Subj": [
        "RT_dec",
        "Accuracy",
        "Subj_idx"
    ],

    # --------------------------------------------------------
    # Adăugăm Stimulus și Response
    # --------------------------------------------------------
    "classification_RT_dec_accuracy_Subj_Stimulus": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Stimulus"
    ],

    "classification_RT_dec_accuracy_Subj_Response": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Response"
    ],

    "classification_RT_dec_accuracy_Subj_Stimulus_Response": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Stimulus",
        "Response"
    ],

    # --------------------------------------------------------
    # Predictori experimentali / manipulări ale sarcinii
    # --------------------------------------------------------
    "classification_RT_dec_accuracy_Subj_Condition": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Condition"
    ],

    "classification_RT_dec_accuracy_Subj_Difficulty": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Difficulty"
    ],

    "classification_RT_dec_accuracy_Subj_Contrast": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Contrast"
    ],

    "classification_RT_dec_accuracy_Subj_Coherence": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Coherence"
    ],

    "classification_RT_dec_accuracy_Subj_Task": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Task"
    ],

    "classification_RT_dec_accuracy_Subj_Training": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Training"
    ],

    # --------------------------------------------------------
    # Structura experimentului
    # --------------------------------------------------------
    "classification_RT_dec_accuracy_Subj_Trial": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Trial"
    ],

    "classification_RT_dec_accuracy_Subj_Block": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Block"
    ],

    "classification_RT_dec_accuracy_Subj_Trial_Block": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Trial",
        "Block"
    ],

    # --------------------------------------------------------
    # Caracteristici individuale
    # --------------------------------------------------------
    "classification_RT_dec_accuracy_Subj_Age": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Age"
    ],

    "classification_RT_dec_accuracy_Subj_Gender": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Gender"
    ],

    "classification_RT_dec_accuracy_Subj_Age_Gender": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Age",
        "Gender"
    ],

    # --------------------------------------------------------
    # Timpi suplimentari
    # --------------------------------------------------------
    "classification_RT_dec_accuracy_Subj_RT_conf": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "RT_conf"
    ],

    "classification_RT_dec_accuracy_Subj_RT_decConf": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "RT_decConf"
    ],

    # --------------------------------------------------------
    # Seturi mai bogate
    # --------------------------------------------------------
    "classification_core_plus_choice": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Stimulus",
        "Response"
    ],

    "classification_core_plus_experimental": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Condition",
        "Difficulty"
    ],

    "classification_core_plus_trial_structure": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Trial",
        "Block"
    ],

    "classification_core_plus_demographics": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Age",
        "Gender"
    ],

    "classification_core_plus_rt_conf": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "RT_conf"
    ],

    "classification_larger_candidate_set": [
        "RT_dec",
        "Accuracy",
        "Subj_idx",
        "Stimulus",
        "Response",
        "Condition",
        "Trial",
        "Block"
    ],
}


simulation_rows = []

for set_name, required_predictors in feature_sets.items():

    n_files_ok = 0
    total_rows_raw = 0
    total_rows_complete = 0

    for file_name in included_files:
        file = RAW_DIR / file_name
        columns = pd.read_csv(file, nrows=0).columns.tolist()

        conf_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Confidence"])

        if conf_col is None:
            continue

        selected_cols = [conf_col]
        missing_predictor = False

        for predictor in required_predictors:
            pred_col = find_column_case_insensitive(columns, COLUMN_GROUPS[predictor])

            if pred_col is None:
                missing_predictor = True
                break

            selected_cols.append(pred_col)

        if missing_predictor:
            continue

        df = pd.read_csv(file, usecols=selected_cols)
        total_rows_raw += len(df)

        conf_class = recodeaza_confidence_3clase(df[conf_col])
        df["confidence_class"] = conf_class

        cols_for_complete = ["confidence_class"]

        for predictor in required_predictors:
            pred_col = find_column_case_insensitive(df.columns, COLUMN_GROUPS[predictor])
            cols_for_complete.append(pred_col)

        df_complete = df.dropna(subset=cols_for_complete)

        if "Accuracy" in required_predictors:
            acc_col = find_column_case_insensitive(df_complete.columns, COLUMN_GROUPS["Accuracy"])
            df_complete[acc_col] = pd.to_numeric(df_complete[acc_col], errors="coerce")
            df_complete = df_complete[df_complete[acc_col].isin([0, 1])]

        if "RT_dec" in required_predictors:
            rt_col = find_column_case_insensitive(df_complete.columns, COLUMN_GROUPS["RT_dec"])
            df_complete[rt_col] = pd.to_numeric(df_complete[rt_col], errors="coerce")
            df_complete = df_complete[df_complete[rt_col] > 0]

        n_complete = len(df_complete)

        if n_complete > 0:
            n_files_ok += 1
            total_rows_complete += n_complete

    simulation_rows.append({
        "feature_set": set_name,
        "required_predictors": ", ".join(required_predictors) if len(required_predictors) > 0 else "niciun predictor",
        "n_files": n_files_ok,
        "total_rows_raw": total_rows_raw,
        "total_rows_complete": total_rows_complete,
        "percent_files_from_included": n_files_ok / len(included_files),
        "percent_complete_rows_from_included_rows": total_rows_complete / confidence_summary["n_rows"].sum(),
    })


simulation_df = pd.DataFrame(simulation_rows)

simulation_df.to_csv(
    REPORTS_DIR / "01b_simulare_seturi_predictori_clasificare.csv",
    index=False
)

print_section("Simulare seturi de predictori pentru clasificare")
print(simulation_df)
