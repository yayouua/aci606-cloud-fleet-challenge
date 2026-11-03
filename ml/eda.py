
# ==========================================================
# ACI606 - CLOUD FLEET CHALLENGE
# Exploratory Data Analysis (EDA)
# ==========================================================

from pathlib import Path

import pandas as pd


# ==========================================================
# 1. CONFIGURACION
# ==========================================================

DATA_PATH = Path("data/training.csv")

REPORTS_DIR = Path("reports")

CLASSES = [
    "NORMAL",
    "WARNING",
    "CRITICAL"
]

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance"
]


# ==========================================================
# 2. CARGAR DATASET
# ==========================================================

def main():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "No existe data/training.csv. "
            "Primero ejecuta generate_dataset.py"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        FEATURES
        + ["robot_id", "fleet_id", "status"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Columnas faltantes: {missing_columns}"
        )

    numeric = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print("\n========== ANALISIS EXPLORATORIO ==========")

    # ======================================================
    # 3. INFORMACION GENERAL
    # ======================================================

    print("\nDIMENSIONES DEL DATASET")

    print("Filas:", len(df))
    print("Columnas:", len(df.columns))

    print("\nFLOTAS")

    print(df["fleet_id"].nunique())

    print("\nROBOTS")

    print(df["robot_id"].nunique())

    # ======================================================
    # 4. DISTRIBUCION DE CLASES
    # ======================================================

    class_counts = (
        df["status"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
    )

    class_distribution = pd.DataFrame({

        "class": CLASSES,

        "count": class_counts.values,

        "percentage": (
            class_counts.values / len(df) * 100
        )

    })

    print("\nDISTRIBUCION DE CLASES")

    print(class_distribution)

    class_distribution.to_csv(
        REPORTS_DIR / "class_distribution.csv",
        index=False
    )

    # ======================================================
    # 5. REGISTROS POR FLOTA
    # ======================================================

    fleet_counts = (
        df.groupby("fleet_id")
        .size()
        .reset_index(name="samples")
    )

    print("\nREGISTROS POR FLOTA")

    print(fleet_counts)

    fleet_counts.to_csv(
        REPORTS_DIR / "fleet_distribution.csv",
        index=False
    )

    # Distribucion de clases por flota
    fleet_classes = pd.crosstab(
        df["fleet_id"],
        df["status"]
    ).reindex(
        columns=CLASSES,
        fill_value=0
    )

    fleet_classes.to_csv(
        REPORTS_DIR / "fleet_class_distribution.csv"
    )

    # ======================================================
    # 6. VALORES FALTANTES
    # ======================================================

    missing_values = pd.DataFrame({

        "column": df.columns,

        "missing_values": [
            df[column].isna().sum()
            for column in df.columns
        ]

    })

    print("\nVALORES FALTANTES")

    print(missing_values)

    missing_values.to_csv(
        REPORTS_DIR / "missing_values.csv",
        index=False
    )

    # ======================================================
    # 7. ESTADISTICAS DESCRIPTIVAS
    # ======================================================

    statistics = numeric.describe().T

    print("\nESTADISTICAS DESCRIPTIVAS")

    print(statistics)

    statistics.to_csv(
        REPORTS_DIR / "summary_statistics.csv"
    )

    # ======================================================
    # 8. PROMEDIOS POR CLASE
    # ======================================================

    class_means = (
        numeric.assign(status=df["status"])
        .groupby("status")
        [FEATURES]
        .mean()
        .reindex(CLASSES)
    )

    print("\nPROMEDIOS POR CLASE")

    print(class_means)

    class_means.to_csv(
        REPORTS_DIR / "class_statistics.csv"
    )

    # ======================================================
    # 9. VALIDACIONES
    # ======================================================

    print("\nVALIDACIONES")

    print(
        "Total de flotas:",
        df["fleet_id"].nunique()
    )

    print(
        "Tres clases presentes:",
        set(CLASSES).issubset(set(df["status"]))
    )

    print(
        "Fleet ID sin valores faltantes:",
        df["fleet_id"].notna().all()
    )

    print(
        "Etiquetas sin valores faltantes:",
        df["status"].notna().all()
    )

    # ======================================================
    # 10. FINALIZACION
    # ======================================================

    print("\nEDA FINALIZADO")

    print("Informes guardados en:", REPORTS_DIR)


if __name__ == "__main__":
    main()
