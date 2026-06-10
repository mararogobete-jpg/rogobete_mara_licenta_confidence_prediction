# 01_explorare_date.py
# Explorarea inițială a fișierelor din Confidence Database.
#
# Scop:
# 1. Inventariem fișierele CSV și README.
# 2. Vedem ce coloane apar cel mai frecvent.
# 3. Analizăm variabila target Confidence.
# 4. Verificăm dacă Confidence este discretă/categorială sau continuă/numerică.
# 5. Stabilim ce fișiere pot fi incluse în analiza principală.

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


# ------------------------------------------------------------
# 2. Funcții simple ajutătoare
# ------------------------------------------------------------

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def find_column_case_insensitive(columns, possible_names):
    """
    Caută o coloană indiferent de litere mari/mici.
    Exemplu: găsește Confidence sau confidence.
    """
    lower_to_original = {str(col).lower(): col for col in columns}

    for name in possible_names:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    return None


def detecteaza_profil_scala(series):
    """
    Analizează empiric seria Confidence pentru a vedea dacă pare:
    - discretă / categorială
    - continuă / numerică
    - ambiguă

    Returnează:
    tip_scala, detaliu_scala
    """

    valid = pd.to_numeric(series, errors="coerce").dropna()

    if len(valid) == 0:
        return "Neconvertibilă", "Lipsă date numerice valide"

    n_unice = valid.nunique()
    min_val = valid.min()
    max_val = valid.max()

    # Verificăm dacă valorile sunt practic întregi:
    # exemple: 1, 2, 3 sau 1.0, 2.0, 3.0
    sunt_intregi = np.all(np.isclose(valid, np.round(valid)))

    # Scale ambigue pentru normalizare automată
    if min_val < 0 or max_val > 100:
        return "Ambiguă", f"Interval neobișnuit: {min_val:g} - {max_val:g}, {n_unice} valori unice"

    # Scale discrete/categoriale
    if n_unice <= 11 and sunt_intregi:
        return "Discretă / categorială", f"Scală {n_unice}-puncte ({min_val:g} - {max_val:g})"

    # Scale continue/numerice
    if min_val >= 0 and max_val <= 1:
        return "Continuă / numerică", f"Continuă [0-1], {n_unice} valori unice"

    if min_val >= 0 and max_val <= 100:
        return "Continuă / numerică", f"Continuă [0-100], {n_unice} valori unice"

    return "Continuă / numerică", f"Alt interval: {min_val:g} - {max_val:g}, {n_unice} valori unice"


def stabileste_includere_target(conf_col, tip_scala):
    """
    Decide dacă fișierul poate fi inclus în analiza principală
    pe baza targetului Confidence.
    """

    if conf_col is None:
        return "exclude_fara_confidence"

    if tip_scala == "Neconvertibilă":
        return "exclude_confidence_neconvertibil"

    if tip_scala == "Ambiguă":
        return "exclude_scala_ambigua"

    return "include_scala_interpretabile"


# ------------------------------------------------------------
# 3. Inventar fișiere
# ------------------------------------------------------------

csv_files = sorted(RAW_DIR.glob("*.csv"))
readme_files = sorted(RAW_DIR.glob("*.txt"))

print_section("Inventar fișiere")
print("Număr fișiere CSV găsite:", len(csv_files))
print("Număr fișiere README găsite:", len(readme_files))

inventar_csv = pd.DataFrame({
    "file": [file.name for file in csv_files]
})

inventar_csv.to_csv(REPORTS_DIR / "01_inventar_fisiere_csv.csv", index=False)


# ------------------------------------------------------------
# 4. Frecvența coloanelor
# ------------------------------------------------------------

column_rows = []
csv_summary_rows = []

for file in csv_files:
    try:
        columns = pd.read_csv(file, nrows=0).columns.tolist()

        for col in columns:
            column_rows.append({
                "file": file.name,
                "column": col
            })

        csv_summary_rows.append({
            "file": file.name,
            "n_columns": len(columns),
            "has_confidence": find_column_case_insensitive(columns, ["Confidence", "confidence"]) is not None,
            "has_accuracy": find_column_case_insensitive(columns, ["Accuracy", "accuracy", "correct", "Correct"]) is not None,
            "has_rt_dec": find_column_case_insensitive(columns, ["RT_dec", "rt_dec", "rt", "RT"]) is not None,
            "has_rt_conf": find_column_case_insensitive(columns, ["RT_conf", "rt_conf"]) is not None,
            "has_condition": find_column_case_insensitive(columns, ["Condition", "condition"]) is not None,
        })

    except Exception as e:
        csv_summary_rows.append({
            "file": file.name,
            "n_columns": np.nan,
            "error": str(e)
        })

columns_df = pd.DataFrame(column_rows)
csv_summary = pd.DataFrame(csv_summary_rows)

column_frequency = (
    columns_df
    .groupby("column")["file"]
    .nunique()
    .reset_index(name="n_files")
    .sort_values("n_files", ascending=False)
    .reset_index(drop=True)
)

csv_summary.to_csv(REPORTS_DIR / "01_rezumat_csv.csv", index=False)
column_frequency.to_csv(REPORTS_DIR / "01_frecventa_coloane.csv", index=False)

print_section("Cele mai frecvente coloane")
print(column_frequency.head(30))


# ------------------------------------------------------------
# 5. Variabile candidate importante
# ------------------------------------------------------------

candidate_names = [
    "Confidence", "confidence",
    "RT_dec", "rt_dec", "rt", "RT",
    "RT_conf", "rt_conf",
    "RT_decConf",
    "Accuracy", "accuracy", "correct", "Correct",
    "Condition", "condition",
    "Difficulty", "difficulty",
    "Contrast", "contrast",
    "Coherence", "coherence",
    "Trial", "trial",
    "Block", "block",
    "Task", "task",
    "Training",
    "Age", "age",
    "Gender",
    "Stimulus", "stimulus",
    "Response", "response",
    "Subj_idx", "subj_idx",
]

candidate_frequency = column_frequency[
    column_frequency["column"].isin(candidate_names)
].copy()

candidate_frequency.to_csv(
    REPORTS_DIR / "01_variabile_candidate.csv",
    index=False
)

print_section("Variabile candidate găsite")
print(candidate_frequency)


# ------------------------------------------------------------
# 6. Analiza targetului Confidence
# ------------------------------------------------------------

target_rows = []

for file in csv_files:
    try:
        columns = pd.read_csv(file, nrows=0).columns.tolist()
        conf_col = find_column_case_insensitive(columns, ["Confidence", "confidence"])

        if conf_col is None:
            tip_scala = "Fără Confidence"
            detaliu_scala = "Nu există coloana Confidence"
            inclusion_status = stabileste_includere_target(conf_col, tip_scala)

            target_rows.append({
                "file": file.name,
                "confidence_col": None,
                "n_rows": np.nan,
                "n_missing": np.nan,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "n_unique": np.nan,
                "n_below_0": np.nan,
                "n_above_1": np.nan,
                "tip_scala_confidence": tip_scala,
                "detaliu_scala_confidence": detaliu_scala,
                "inclusion_status": inclusion_status,
            })
            continue

        df = pd.read_csv(file, usecols=[conf_col])
        conf = pd.to_numeric(df[conf_col], errors="coerce")

        n_rows = len(conf)
        n_missing = conf.isna().sum()
        valid = conf.dropna()

        tip_scala, detaliu_scala = detecteaza_profil_scala(conf)
        inclusion_status = stabileste_includere_target(conf_col, tip_scala)

        if len(valid) == 0:
            target_rows.append({
                "file": file.name,
                "confidence_col": conf_col,
                "n_rows": n_rows,
                "n_missing": n_missing,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "n_unique": 0,
                "n_below_0": np.nan,
                "n_above_1": np.nan,
                "tip_scala_confidence": tip_scala,
                "detaliu_scala_confidence": detaliu_scala,
                "inclusion_status": inclusion_status,
            })
            continue

        target_rows.append({
            "file": file.name,
            "confidence_col": conf_col,
            "n_rows": n_rows,
            "n_missing": n_missing,
            "min": valid.min(),
            "max": valid.max(),
            "mean": valid.mean(),
            "std": valid.std(),
            "median": valid.median(),
            "q25": valid.quantile(0.25),
            "q75": valid.quantile(0.75),
            "n_unique": valid.nunique(),
            "n_below_0": (valid < 0).sum(),
            "n_above_1": (valid > 1).sum(),
            "tip_scala_confidence": tip_scala,
            "detaliu_scala_confidence": detaliu_scala,
            "inclusion_status": inclusion_status,
        })

    except Exception as e:
        target_rows.append({
            "file": file.name,
            "confidence_col": None,
            "error": str(e),
            "n_rows": np.nan,
            "n_missing": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "n_unique": np.nan,
            "n_below_0": np.nan,
            "n_above_1": np.nan,
            "tip_scala_confidence": "Eroare citire",
            "detaliu_scala_confidence": str(e),
            "inclusion_status": "exclude_eroare_citire",
        })


target_summary = pd.DataFrame(target_rows)

target_summary.to_csv(
    REPORTS_DIR / "01_target_confidence_summary_cu_includere.csv",
    index=False
)

print_section("Analiza targetului Confidence/confidence")

print("Număr fișiere totale:", len(target_summary))
print("Număr fișiere cu target confidence:", target_summary["confidence_col"].notna().sum())
print("Număr fișiere fără target confidence:", target_summary["confidence_col"].isna().sum())

print()
print("Profilul scalelor Confidence:")
print(target_summary["tip_scala_confidence"].value_counts(dropna=False))

print()
print("Cele mai frecvente detalii ale scalei Confidence:")
print(target_summary["detaliu_scala_confidence"].value_counts(dropna=False).head(20))


# ------------------------------------------------------------
# 7. Statistici descriptive pentru Confidence brut
# ------------------------------------------------------------

numeric_cols_for_target = [
    "n_rows", "n_missing", "min", "max", "mean", "std",
    "median", "q25", "q75", "n_unique", "n_below_0", "n_above_1"
]

print_section("Statistici descriptive pentru Confidence brut, la nivel de experimente")
print(target_summary[numeric_cols_for_target].describe().T)


# ------------------------------------------------------------
# 8. Simulare includere/excludere pe baza targetului
# ------------------------------------------------------------

inclusion_summary = (
    target_summary
    .groupby("inclusion_status")
    .agg(
        n_files=("file", "count"),
        total_rows=("n_rows", "sum")
    )
    .reset_index()
    .sort_values("n_files", ascending=False)
)

inclusion_summary.to_csv(
    REPORTS_DIR / "01_rezumat_includere_excludere_confidence.csv",
    index=False
)

n_files_total = len(target_summary)
n_files_included = (target_summary["inclusion_status"] == "include_scala_interpretabile").sum()
n_files_excluded = n_files_total - n_files_included

rows_total = target_summary["n_rows"].sum(skipna=True)
rows_included = target_summary.loc[
    target_summary["inclusion_status"] == "include_scala_interpretabile",
    "n_rows"
].sum(skipna=True)

rows_excluded = rows_total - rows_included

print_section("Simulare includere/excludere pe baza targetului Confidence")
print(inclusion_summary)

print()
print("Fișiere totale:", n_files_total)
print("Fișiere incluse:", n_files_included)
print("Fișiere excluse:", n_files_excluded)
print("Procent fișiere incluse:", round(n_files_included / n_files_total * 100, 2), "%")

print()
print("Rânduri totale cu informație numerică disponibilă:", f"{rows_total:,.0f}")
print("Rânduri incluse:", f"{rows_included:,.0f}")
print("Rânduri excluse:", f"{rows_excluded:,.0f}")
print("Procent rânduri incluse:", round(rows_included / rows_total * 100, 2), "%")

print()
print("Fișiere excluse din analiza principală:")
excluded_files = target_summary.loc[
    target_summary["inclusion_status"] != "include_scala_interpretabile",
    [
        "file",
        "inclusion_status",
        "n_rows",
        "min",
        "max",
        "mean",
        "tip_scala_confidence",
        "detaliu_scala_confidence",
    ]
]

print(excluded_files)


# ------------------------------------------------------------
# 9. Figuri simple pentru scală Confidence
# ------------------------------------------------------------

scale_type_counts = (
    target_summary["tip_scala_confidence"]
    .value_counts()
    .reset_index()
)

scale_type_counts.columns = ["tip_scala_confidence", "n_files"]

plt.figure(figsize=(8, 5))
plt.bar(scale_type_counts["tip_scala_confidence"], scale_type_counts["n_files"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Număr fișiere")
plt.title("Tipuri de scală pentru Confidence")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_tipuri_scala_confidence.png", dpi=150)
plt.close()


detail_counts = (
    target_summary["detaliu_scala_confidence"]
    .value_counts()
    .head(15)
    .reset_index()
)

detail_counts.columns = ["detaliu_scala_confidence", "n_files"]

plt.figure(figsize=(10, 5))
plt.bar(detail_counts["detaliu_scala_confidence"], detail_counts["n_files"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Număr fișiere")
plt.title("Cele mai frecvente scale observate pentru Confidence")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_detalii_scala_confidence.png", dpi=150)
plt.close()


