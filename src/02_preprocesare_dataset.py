# 02_preprocesare_dataset.py
# Preprocesarea datasetului principal pentru clasificare.
#
# Scop:
# 1. Folosim doar fișierele cu Confidence interpretabil.
# 2. Transformăm Confidence în 3 clase:
#    0 = low confidence
#    1 = medium confidence
#    2 = high confidence
# 3. Construim uid_participant = experiment_id + "_" + subj_id.
# 4. Curățăm RT_dec:
#    - păstrăm doar 0.1 <= RT_dec <= 5.0 secunde
#    - calculăm z-score pe log(RT_dec), separat per uid_participant
#    - păstrăm doar |z| <= 3
# 5. Construim lag_rt_dec_log_z folosind shift(1) per uid_participant.
# 6. Salvăm datasetul final pentru clasificare.

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

PROCESSED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print("Folder proiect:", PROJECT_DIR)
print("Folder date raw:", RAW_DIR)
print("Folder date processed:", PROCESSED_DIR)
print("Folder rapoarte:", REPORTS_DIR)


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
    """
    Recodifică Confidence în 3 clase ținând cont de treptele distincte
    ale scalei din fiecare experiment.

    Important:
    - fiecare fișier corespunde unui experiment;
    - normalizăm mai întâi confidence în [0, 1];
    - apoi împărțim valorile distincte ale scalei în low / medium / high,
      nu intervalul numeric [0, 1] în treimi egale.

    Exemple:
    - scală cu 3 trepte: low, medium, high
    - scală cu 4 trepte: low, medium, medium, high
    - scală cu 6 trepte: low, low, medium, medium, high, high
    """

    conf_01 = normalize_minmax(series)
    conf_round = conf_01.round(6)

    values = np.sort(conf_round.dropna().unique())
    n = len(values)

    confidence_class = pd.Series(np.nan, index=series.index, dtype="float")

    if n == 0:
        return confidence_class

    if n == 1:
        value_to_class = {
            values[0]: 1
        }

    elif n == 2:
        value_to_class = {
            values[0]: 0,
            values[1]: 2
        }

    else:
        n_low = int(np.ceil(n / 3))
        n_high = int(np.ceil(n / 3))
        n_medium = n - n_low - n_high

        # Pentru scale mici, ne asigurăm că există zonă de mijloc.
        # Exemplu: 4 trepte -> low, medium, medium, high.
        if n_medium < 1:
            n_low = 1
            n_high = 1
            n_medium = n - 2

        low_values = values[:n_low]
        medium_values = values[n_low:n_low + n_medium]
        high_values = values[n_low + n_medium:]

        value_to_class = {}

        for v in low_values:
            value_to_class[v] = 0

        for v in medium_values:
            value_to_class[v] = 1

        for v in high_values:
            value_to_class[v] = 2

    confidence_class = conf_round.map(value_to_class).astype("float")

    return confidence_class


# ------------------------------------------------------------
# 3. Coloane posibile
# ------------------------------------------------------------

COLUMN_GROUPS = {
    "Confidence": ["Confidence", "confidence"],
    "RT_dec": ["RT_dec", "rt_dec", "rt", "RT"],
    "Accuracy": ["Accuracy", "accuracy", "correct", "Correct"],
    "Subj_idx": ["Subj_idx", "subj_idx"],
    "Trial": ["Trial", "trial"],
}


# ------------------------------------------------------------
# 4. Reguli de curățare RT
# ------------------------------------------------------------

RT_MIN_SECONDS = 0.1
RT_MAX_SECONDS = 5.0
RT_Z_MAX_ABS = 3.0


# ------------------------------------------------------------
# 5. Încărcăm lista fișierelor incluse din 01
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

print_section("Fișiere incluse după analiza targetului")
print("Număr fișiere incluse:", len(included_files))


# ------------------------------------------------------------
# 6. Construim datasetul principal
# ------------------------------------------------------------

dataset_parts = []
log_rows = []

for file_name in included_files:
    file = RAW_DIR / file_name

    try:
        columns = pd.read_csv(file, nrows=0).columns.tolist()

        conf_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Confidence"])
        rt_col = find_column_case_insensitive(columns, COLUMN_GROUPS["RT_dec"])
        acc_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Accuracy"])
        subj_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Subj_idx"])
        trial_col = find_column_case_insensitive(columns, COLUMN_GROUPS["Trial"])

        if conf_col is None or rt_col is None or acc_col is None or subj_col is None:
            log_rows.append({
                "file": file_name,
                "status": "skip_missing_required_columns",
                "confidence_col": conf_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "subj_col": subj_col,
                "trial_col": trial_col,
                "rows_before": np.nan,
                "rows_after_basic": 0,
                "rows_after_rt_range": 0,
                "rows_after_z_filter": 0,
                "rows_after_lag": 0,
            })
            continue

        usecols = [conf_col, rt_col, acc_col, subj_col]

        if trial_col is not None:
            usecols.append(trial_col)

        df = pd.read_csv(file, usecols=usecols, low_memory=False)

        rename_map = {
            conf_col: "confidence",
            rt_col: "rt_dec",
            acc_col: "accuracy",
            subj_col: "subj_id",
        }

        if trial_col is not None:
            rename_map[trial_col] = "trial"

        df = df.rename(columns=rename_map)

        experiment_id = Path(file_name).stem
        rows_before = len(df)

        # Păstrăm ordinea originală din fișier ca rezervă pentru ordonarea trialurilor.
        df["row_order"] = np.arange(len(df))

        # Conversii numerice
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["rt_dec"] = pd.to_numeric(df["rt_dec"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

        if "trial" in df.columns:
            df["trial_order"] = pd.to_numeric(df["trial"], errors="coerce")
            df["trial_order"] = df["trial_order"].fillna(df["row_order"])
        else:
            df["trial_order"] = df["row_order"]

        # Target: confidence_01 + confidence_class
        df["confidence_01"] = normalize_minmax(df["confidence"])
        df["confidence_class"] = recodeaza_confidence_3clase(df["confidence"])

        # Identificatori
        df["experiment_id"] = experiment_id
        df["subj_id"] = df["subj_id"].astype(str)
        df["uid_participant"] = df["experiment_id"] + "_" + df["subj_id"]

        # Filtrare de bază
        df = df.dropna(subset=[
            "confidence_01",
            "confidence_class",
            "rt_dec",
            "accuracy",
            "subj_id",
            "uid_participant",
        ])

        df = df[df["rt_dec"] > 0]
        df = df[df["accuracy"].isin([0, 1])]

        rows_after_basic = len(df)

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_valid_rows",
                "confidence_col": conf_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "subj_col": subj_col,
                "trial_col": trial_col,
                "rows_before": rows_before,
                "rows_after_basic": rows_after_basic,
                "rows_after_rt_range": 0,
                "rows_after_z_filter": 0,
                "rows_after_lag": 0,
            })
            continue

        # ----------------------------------------------------
        # Pasul 1 RT: păstrăm doar RT între 0.1 și 5 secunde
        # ----------------------------------------------------

        df = df[
            (df["rt_dec"] >= RT_MIN_SECONDS)
            & (df["rt_dec"] <= RT_MAX_SECONDS)
        ].copy()

        rows_after_rt_range = len(df)

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_rows_after_rt_range",
                "confidence_col": conf_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "subj_col": subj_col,
                "trial_col": trial_col,
                "rows_before": rows_before,
                "rows_after_basic": rows_after_basic,
                "rows_after_rt_range": rows_after_rt_range,
                "rows_after_z_filter": 0,
                "rows_after_lag": 0,
            })
            continue

        # ----------------------------------------------------
        # Pasul 2 RT: log(RT) + z-score per uid_participant
        # ----------------------------------------------------

        df["rt_dec_clean"] = df["rt_dec"]
        df["rt_dec_log"] = np.log(df["rt_dec_clean"])

        mean_per_participant = df.groupby("uid_participant")["rt_dec_log"].transform("mean")
        std_per_participant = df.groupby("uid_participant")["rt_dec_log"].transform("std")

        df["rt_dec_log_z"] = (df["rt_dec_log"] - mean_per_participant) / std_per_participant

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["rt_dec_log_z"])

        df = df[df["rt_dec_log_z"].abs() <= RT_Z_MAX_ABS].copy()

        rows_after_z_filter = len(df)

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_rows_after_z_filter",
                "confidence_col": conf_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "subj_col": subj_col,
                "trial_col": trial_col,
                "rows_before": rows_before,
                "rows_after_basic": rows_after_basic,
                "rows_after_rt_range": rows_after_rt_range,
                "rows_after_z_filter": rows_after_z_filter,
                "rows_after_lag": 0,
            })
            continue

        # ----------------------------------------------------
        # Variabila lag: RT-ul de la trialul anterior
        # ----------------------------------------------------
        # Important:
        # shift(1) se face STRICT în interiorul fiecărui uid_participant.

        df = df.sort_values(["uid_participant", "trial_order", "row_order"])

        df["lag_rt_dec_log_z"] = (
            df.groupby("uid_participant")["rt_dec_log_z"].shift(1)
        )

        df = df.dropna(subset=["lag_rt_dec_log_z"]).copy()

        rows_after_lag = len(df)

        if len(df) == 0:
            log_rows.append({
                "file": file_name,
                "status": "skip_no_rows_after_lag",
                "confidence_col": conf_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "subj_col": subj_col,
                "trial_col": trial_col,
                "rows_before": rows_before,
                "rows_after_basic": rows_after_basic,
                "rows_after_rt_range": rows_after_rt_range,
                "rows_after_z_filter": rows_after_z_filter,
                "rows_after_lag": rows_after_lag,
            })
            continue

        # Interacțiune RT x accuracy
        df["rt_x_acc"] = df["rt_dec_log_z"] * df["accuracy"]

        # Tipuri finale
        df["confidence_class"] = df["confidence_class"].astype(int)
        df["accuracy"] = df["accuracy"].astype(int)

        keep_cols = [
            "confidence_class",
            "confidence_01",
            "accuracy",
            "rt_dec_clean",
            "rt_dec_log",
            "rt_dec_log_z",
            "rt_x_acc",
            "lag_rt_dec_log_z",
            "experiment_id",
            "subj_id",
            "uid_participant",
            "trial_order",
        ]

        df = df[keep_cols]

        dataset_parts.append(df)

        log_rows.append({
            "file": file_name,
            "status": "included_main_classification",
            "confidence_col": conf_col,
            "rt_col": rt_col,
            "accuracy_col": acc_col,
            "subj_col": subj_col,
            "trial_col": trial_col,
            "rows_before": rows_before,
            "rows_after_basic": rows_after_basic,
            "rows_after_rt_range": rows_after_rt_range,
            "rows_after_z_filter": rows_after_z_filter,
            "rows_after_lag": rows_after_lag,
        })

    except Exception as e:
        log_rows.append({
            "file": file_name,
            "status": "error",
            "error": str(e),
            "rows_before": np.nan,
            "rows_after_basic": 0,
            "rows_after_rt_range": 0,
            "rows_after_z_filter": 0,
            "rows_after_lag": 0,
        })


if len(dataset_parts) == 0:
    raise ValueError("Nu s-a putut construi datasetul principal pentru clasificare.")

dataset_main = pd.concat(dataset_parts, ignore_index=True)
log_df = pd.DataFrame(log_rows)


# ------------------------------------------------------------
# 7. Salvăm datasetul și logul
# ------------------------------------------------------------

dataset_path = PROCESSED_DIR / "dataset_main_classification.csv"
log_path = REPORTS_DIR / "02_log_preprocesare_dataset_main_classification.csv"
summary_path = REPORTS_DIR / "02_rezumat_dataset_main_classification.csv"
class_dist_path = REPORTS_DIR / "02_distributie_confidence_class_final.csv"

dataset_main.to_csv(dataset_path, index=False)
log_df.to_csv(log_path, index=False)


# ------------------------------------------------------------
# 8. Rezumat dataset final
# ------------------------------------------------------------

print_section("Dataset principal pentru clasificare salvat")
print("Fișier:", dataset_path)
print("Shape:", dataset_main.shape)
print("Număr experimente:", dataset_main["experiment_id"].nunique())
print("Număr participanți unici:", dataset_main["uid_participant"].nunique())

print()
print("Status fișiere în log:")
print(log_df["status"].value_counts())


print_section("Pierderi de observații în preprocesare")

total_before = log_df["rows_before"].sum(skipna=True)
total_basic = log_df["rows_after_basic"].sum(skipna=True)
total_rt_range = log_df["rows_after_rt_range"].sum(skipna=True)
total_z = log_df["rows_after_z_filter"].sum(skipna=True)
total_lag = log_df["rows_after_lag"].sum(skipna=True)

loss_summary = pd.DataFrame({
    "etapa": [
        "rows_before",
        "after_basic_filter",
        "after_rt_0.1_5",
        "after_z_abs_3",
        "after_lag_dropna",
    ],
    "n_rows": [
        total_before,
        total_basic,
        total_rt_range,
        total_z,
        total_lag,
    ]
})

loss_summary["percent_from_initial"] = loss_summary["n_rows"] / total_before

print(loss_summary)


# ------------------------------------------------------------
# 9. Distribuția targetului final
# ------------------------------------------------------------

print_section("Distribuția finală a targetului confidence_class")

class_distribution = (
    dataset_main["confidence_class"]
    .value_counts()
    .sort_index()
    .reset_index()
)

class_distribution.columns = ["confidence_class", "n_rows"]
class_distribution["percent"] = (
    class_distribution["n_rows"] / class_distribution["n_rows"].sum()
)

class_distribution["label"] = class_distribution["confidence_class"].map({
    0: "low confidence",
    1: "medium confidence",
    2: "high confidence",
})

print(class_distribution)

class_distribution.to_csv(class_dist_path, index=False)


# ------------------------------------------------------------
# 10. Statistici descriptive
# ------------------------------------------------------------

print_section("Statistici descriptive dataset final")

desc_cols = [
    "confidence_class",
    "confidence_01",
    "accuracy",
    "rt_dec_clean",
    "rt_dec_log",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

desc_main = dataset_main[desc_cols].describe().T
print(desc_main)

print()
print("Distribuție accuracy:")
print(dataset_main["accuracy"].value_counts(normalize=True).sort_index())

print()
print("Număr observații per experiment:")
print(dataset_main.groupby("experiment_id").size().describe())

print()
print("Număr observații per participant:")
print(dataset_main.groupby("uid_participant").size().describe())

print()
print("Verificare valori lipsă:")
print(dataset_main.isna().sum())


# ------------------------------------------------------------
# 11. Rezumat pentru reports
# ------------------------------------------------------------

summary_df = pd.DataFrame([
    {
        "dataset": "dataset_main_classification",
        "n_rows": len(dataset_main),
        "n_experiments": dataset_main["experiment_id"].nunique(),
        "n_participants": dataset_main["uid_participant"].nunique(),
        "target": "confidence_class",
        "classes": "0=low, 1=medium, 2=high",
        "features": "accuracy, rt_dec_log_z, rt_x_acc, lag_rt_dec_log_z",
        "normalization": "min-max observed per experiment",
        "class_rule": "scale-aware recoding based on distinct confidence_01 values within each experiment",
        "rt_rule": "keep 0.1s <= RT_dec <= 5s; then keep |RT log z-score per participant| <= 3",
        "lag_rule": "lag_rt_dec_log_z created with groupby(uid_participant).shift(1)",
        "participant_rule": "uid_participant = experiment_id + subj_id",
    }
])

summary_df.to_csv(summary_path, index=False)
loss_summary.to_csv(REPORTS_DIR / "02_pierderi_observatii_preprocesare.csv", index=False)

print_section("Rezumat dataset principal")
print(summary_df)