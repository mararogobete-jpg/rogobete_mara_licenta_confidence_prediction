# 02a_grafice_distributie_variabile.py
# Grafice exploratorii relevante pentru datasetul final.
#
# Scop:
# 1. Vizualizăm distribuția claselor de confidence.
# 2. Vizualizăm transformarea timpului de reacție:
#    RT_dec_clean -> rt_dec_log -> rt_dec_log_z.
# 3. Analizăm relația dintre accuracy și clasele de confidence.
# 4. Analizăm distribuția RT standardizat pe clase de confidence.
# 5. Calculăm matricea de corelație Spearman pentru target și predictorii finali.
#
# Notă:
# Acest script se rulează după 02_preprocesare_dataset.py,
# deoarece folosește datasetul final:
# data/processed/dataset_main_classification.csv

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Setări generale
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "processed" / "dataset_main_classification.csv"

FIGURES_DIR = PROJECT_DIR / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET = "confidence_class"

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}

FEATURES = [
    "accuracy",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

CORR_VARS = [
    TARGET,
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


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Grafic salvat:", path)


# ------------------------------------------------------------
# 3. Încărcarea datasetului final
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Nu găsesc data/processed/dataset_main_classification.csv. "
        "Rulează mai întâi 02_preprocesare_dataset.py."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

needed_cols = [
    TARGET,
    "confidence_01",
    "accuracy",
    "rt_dec_clean",
    "rt_dec_log",
    "rt_dec_log_z",
    "rt_x_acc",
    "lag_rt_dec_log_z",
]

missing_cols = [col for col in needed_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Lipsesc coloanele necesare: {missing_cols}")

df = df.dropna(subset=needed_cols).copy()
df[TARGET] = df[TARGET].astype(int)

print_section("Dataset final încărcat")
print("Dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Coloane folosite:", needed_cols)


# ------------------------------------------------------------
# 4. Distribuția timpului de reacție înainte și după transformare
# ------------------------------------------------------------

print_section("Distribuția timpului de reacție înainte și după transformare")

# 4.1 RT_dec_clean - după filtrarea 0.1s <= RT_dec <= 5s
plt.figure(figsize=(8, 5))

plt.hist(
    df["rt_dec_clean"],
    bins=80,
    edgecolor="black",
    alpha=0.8,
)

plt.title("Distribuția timpului de reacție după filtrare")
plt.xlabel("RT_dec_clean (secunde)")
plt.ylabel("Frecvență")

save_fig(FIGURES_DIR / "02a_00a_distributie_rt_dec_clean.png")


# 4.2 rt_dec_log - după log-transformare
plt.figure(figsize=(8, 5))

plt.hist(
    df["rt_dec_log"],
    bins=80,
    edgecolor="black",
    alpha=0.8,
)

plt.title("Distribuția timpului de reacție după log-transformare")
plt.xlabel("log(RT_dec_clean)")
plt.ylabel("Frecvență")

save_fig(FIGURES_DIR / "02a_00b_distributie_rt_dec_log.png")


# 4.3 rt_dec_log_z - după standardizare pe participant
plt.figure(figsize=(8, 5))

plt.hist(
    df["rt_dec_log_z"],
    bins=80,
    edgecolor="black",
    alpha=0.8,
)

plt.title("Distribuția timpului de reacție după standardizare pe participant")
plt.xlabel("rt_dec_log_z")
plt.ylabel("Frecvență")

save_fig(FIGURES_DIR / "02a_00c_distributie_rt_dec_log_z.png")


# ------------------------------------------------------------
# 5. Distribuția targetului confidence_class
# ------------------------------------------------------------

print_section("Distribuția claselor de confidence")

target_counts = (
    df[TARGET]
    .value_counts()
    .sort_index()
    .reset_index()
)

target_counts.columns = ["confidence_class", "n_rows"]
target_counts["percent"] = target_counts["n_rows"] / target_counts["n_rows"].sum()
target_counts["label"] = target_counts["confidence_class"].map(CLASS_NAMES)

print(target_counts)

target_counts.to_csv(
    REPORTS_DIR / "02a_distributie_confidence_class.csv",
    index=False,
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    target_counts["label"],
    target_counts["n_rows"],
)

plt.title("Distribuția claselor de confidence")
plt.xlabel("Clasa confidence")
plt.ylabel("Număr observații")

for bar, percent in zip(bars, target_counts["percent"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{percent:.1%}",
        ha="center",
        va="bottom",
    )

save_fig(FIGURES_DIR / "02a_01_distributie_confidence_class.png")


# ------------------------------------------------------------
# 6. Proporția de răspunsuri corecte pe clase de confidence
# ------------------------------------------------------------

print_section("Proporția de răspunsuri corecte pe clase de confidence")

accuracy_by_class = (
    df
    .groupby(TARGET)["accuracy"]
    .mean()
    .reset_index()
)

accuracy_by_class["label"] = accuracy_by_class[TARGET].map(CLASS_NAMES)

print(accuracy_by_class)

accuracy_by_class.to_csv(
    REPORTS_DIR / "02a_accuracy_medie_pe_clase.csv",
    index=False,
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    accuracy_by_class["label"],
    accuracy_by_class["accuracy"],
)

plt.title("Proporția de răspunsuri corecte pe clase de confidence")
plt.xlabel("Clasa confidence")
plt.ylabel("Proporție răspunsuri corecte")
plt.ylim(0, 1)

for bar, value in zip(bars, accuracy_by_class["accuracy"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{value:.1%}",
        ha="center",
        va="bottom",
    )

save_fig(FIGURES_DIR / "02a_02_accuracy_pe_clase_confidence.png")


# ------------------------------------------------------------
# 7. Distribuția RT_dec_log_z pe clase de confidence
# ------------------------------------------------------------

print_section("Distribuția RT_dec_log_z pe clase de confidence")

rt_by_class = [
    df.loc[df[TARGET] == cls, "rt_dec_log_z"].dropna()
    for cls in CLASS_LABELS
]

plt.figure(figsize=(7, 5))

plt.boxplot(
    rt_by_class,
    labels=[CLASS_NAMES[cls] for cls in CLASS_LABELS],
    showfliers=False,
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Distribuția timpului de reacție standardizat pe clase de confidence")
plt.xlabel("Clasa confidence")
plt.ylabel("rt_dec_log_z")

save_fig(FIGURES_DIR / "02a_03_boxplot_rt_dec_log_z_pe_clase.png")


# ------------------------------------------------------------
# 8. Matrice de corelație Spearman
# ------------------------------------------------------------

print_section("Matrice de corelație Spearman")

corr = df[CORR_VARS].corr(method="spearman")

print(corr)

corr.to_csv(
    REPORTS_DIR / "02a_matrice_spearman_predictori.csv",
    index=True,
)

display_labels = [
    "Confidence class",
    "Accuracy",
    "RT_dec log z",
    "RT × Accuracy",
    "Lag RT_dec log z",
]

plt.figure(figsize=(9, 7))

im = plt.imshow(corr.values, vmin=-1, vmax=1)

plt.colorbar(im, label="Spearman rho")

plt.xticks(
    ticks=np.arange(len(display_labels)),
    labels=display_labels,
    rotation=35,
    ha="right",
)

plt.yticks(
    ticks=np.arange(len(display_labels)),
    labels=display_labels,
)

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(
            j,
            i,
            f"{corr.values[i, j]:.3f}",
            ha="center",
            va="center",
        )

plt.title("Matrice de corelație Spearman - target și predictori")

save_fig(FIGURES_DIR / "02a_04_matrice_spearman_predictori.png")


# ------------------------------------------------------------
# 9. Rezumat descriptiv
# ------------------------------------------------------------

print_section("Rezumat descriptiv variabile principale")

summary = df[
    [
        TARGET,
        "confidence_01",
        "accuracy",
        "rt_dec_clean",
        "rt_dec_log",
        "rt_dec_log_z",
        "rt_x_acc",
        "lag_rt_dec_log_z",
    ]
].describe().T

print(summary)

summary.to_csv(
    REPORTS_DIR / "02a_rezumat_descriptiv_variabile.csv",
    index=True,
)


# ------------------------------------------------------------
# 10. Final
# ------------------------------------------------------------

print_section("Final")

print("Grafice relevante salvate în folderul:", FIGURES_DIR)

print()
print("Grafice generate:")
print("1. 02a_00a_distributie_rt_dec_clean.png")
print("2. 02a_00b_distributie_rt_dec_log.png")
print("3. 02a_00c_distributie_rt_dec_log_z.png")
print("4. 02a_01_distributie_confidence_class.png")
print("5. 02a_02_accuracy_pe_clase_confidence.png")
print("6. 02a_03_boxplot_rt_dec_log_z_pe_clase.png")
print("7. 02a_04_matrice_spearman_predictori.png")

print()
print("Rapoarte generate:")
print("1. 02a_distributie_confidence_class.csv")
print("2. 02a_accuracy_medie_pe_clase.csv")
print("3. 02a_matrice_spearman_predictori.csv")
print("4. 02a_rezumat_descriptiv_variabile.csv")