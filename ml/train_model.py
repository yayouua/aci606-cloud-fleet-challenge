
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "training.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ---------------------------------------------------------
# 2. CONFIGURACIÓN DEL ENTRENAMIENTO
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.25

TARGET_CANDIDATES = [
    "status",
    "label",
    "target",
    "estado",
    "risk_level",
    "condition",
]

FEATURE_CANDIDATES = [
    "x",
    "y",
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]


# ---------------------------------------------------------
# 3. CARGAR Y VALIDAR EL DATASET
# ---------------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {DATA_PATH}\n"
            "Primero debes completar el paso 11 y generar "
            "data/training.csv."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if "fleet_id" not in df.columns:
        raise ValueError(
            "El dataset debe contener la columna fleet_id "
            "para separar las flotas."
        )

    target_column = next(
        (column for column in TARGET_CANDIDATES if column in df.columns),
        None,
    )

    if target_column is None:
        raise ValueError(
            "No se encontró la columna objetivo. "
            f"Se buscaron: {TARGET_CANDIDATES}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    feature_columns = [
        column
        for column in FEATURE_CANDIDATES
        if column in df.columns
    ]

    if len(feature_columns) < 2:
        raise ValueError(
            "No hay suficientes variables predictoras. "
            f"Se encontraron: {feature_columns}"
        )

    # Convertimos las variables predictoras a formato numérico.
    X = df[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Eliminamos columnas completamente vacías.
    feature_columns = [
        column
        for column in feature_columns
        if not X[column].isna().all()
    ]
    X = X[feature_columns]

    if len(feature_columns) < 2:
        raise ValueError(
            "Menos de dos variables tienen datos numéricos válidos."
        )

    # Eliminamos filas sin clase o sin identificación de flota.
    y = df[target_column].astype("string").str.strip()
    groups = df["fleet_id"].astype("string").str.strip()

    valid_rows = (
        y.notna()
        & groups.notna()
        & (y != "")
        & (groups != "")
    )

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].astype(str).reset_index(drop=True)
    groups = groups.loc[valid_rows].astype(str).reset_index(drop=True)

    if len(X) < 10:
        raise ValueError(
            f"Solo quedan {len(X)} registros válidos. "
            "Genera más datos antes de entrenar."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Se necesitan al menos dos clases diferentes "
            "en la columna objetivo."
        )

    if groups.nunique() < 2:
        raise ValueError(
            "Se necesitan al menos dos fleet_id diferentes "
            "para evaluar flotas no vistas."
        )

    print("\nDATASET CARGADO")
    print("------------------------------")
    print(f"Registros válidos: {len(X)}")
    print(f"Flotas: {groups.nunique()}")
    print(f"Clases: {sorted(y.unique())}")
    print(f"Variables predictoras: {feature_columns}")
    print(f"Variable objetivo: {target_column}")

    return X, y, groups, target_column, feature_columns


# ---------------------------------------------------------
# 4. SEPARAR FLOTAS DE ENTRENAMIENTO Y PRUEBA
# ---------------------------------------------------------

def split_by_fleet(X, y, groups):
    """
    Busca una separación donde no se compartan fleet_id
    entre entrenamiento y prueba.

    Intenta mantener todas las clases en entrenamiento y,
    cuando sea posible, también en prueba.
    """

    all_classes = set(y.unique())
    best_split = None
    best_test_coverage = -1

    for seed in range(RANDOM_STATE, RANDOM_STATE + 200):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=seed,
        )

        train_indices, test_indices = next(
            splitter.split(X, y, groups)
        )

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        train_classes = set(y_train.unique())
        test_classes = set(y_test.unique())

        # El entrenamiento debe conocer todas las clases.
        if train_classes != all_classes:
            continue

        # Preferimos una prueba con todas las clases.
        coverage = len(test_classes)

        if coverage > best_test_coverage:
            best_test_coverage = coverage
            best_split = (train_indices, test_indices)

        if test_classes == all_classes:
            break

    if best_split is None:
        raise ValueError(
            "No se pudo construir una separación por flota "
            "que contenga todas las clases en entrenamiento. "
            "Revisa la distribución de clases entre fleet_id "
            "o genera más datos."
        )

    train_indices, test_indices = best_split

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    groups_train = groups.iloc[train_indices]
    groups_test = groups.iloc[test_indices]

    overlap = set(groups_train) & set(groups_test)

    if overlap:
        raise RuntimeError(
            f"Error: hay flotas compartidas entre entrenamiento "
            f"y prueba: {overlap}"
        )

    if set(y_test.unique()) != all_classes:
        warnings.warn(
            "La prueba no contiene todas las clases. "
            "Revisa la distribución de clases por flota."
        )

    print("\nSEPARACIÓN POR FLOTAS")
    print("------------------------------")
    print(f"Registros entrenamiento: {len(X_train)}")
    print(f"Registros prueba: {len(X_test)}")
    print(f"Flotas entrenamiento: {sorted(groups_train.unique())}")
    print(f"Flotas prueba: {sorted(groups_test.unique())}")
    print("Flotas compartidas: ninguna")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    )


# ---------------------------------------------------------
# 5. DEFINIR LOS DOS MODELOS
# ---------------------------------------------------------

def build_models():
    models = {
        "RandomForest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "LogisticRegression": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# ---------------------------------------------------------
# 6. ENTRENAR Y EVALUAR
# ---------------------------------------------------------

def train_and_evaluate(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
):
    results = []
    trained_models = {}

    for model_name, pipeline in models.items():
        print(f"\nEntrenando: {model_name}")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "model": model_name,
            "f1_macro": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision_macro": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        results.append(metrics)
        trained_models[model_name] = pipeline

        print(f"F1 macro: {metrics['f1_macro']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by="f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    return comparison, trained_models


# ---------------------------------------------------------
# 7. GUARDAR RESULTADOS Y MATRIZ DE CONFUSIÓN
# ---------------------------------------------------------

def save_results(
    comparison,
    trained_models,
    X_test,
    y_test,
    groups_test,
    target_column,
    feature_columns,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Seleccionamos el modelo con mayor F1 macro.
    best_model_name = comparison.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]

    predictions = best_pipeline.predict(X_test)

    labels = sorted(set(y_test) | set(predictions))

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    # Comparación de modelos.
    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Matriz de confusión del mejor modelo.
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    matrix_df.index.name = "real_predicho"
    matrix_df.columns.name = "prediccion"

    matrix_path = REPORT_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    # Reporte por clase.
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )

    report_path = REPORT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Guardamos el pipeline para utilizarlo desde la API.
    joblib.dump(best_pipeline, MODEL_PATH)

    best_metrics = comparison.iloc[0].to_dict()

    metadata = {
        "model_name": best_model_name,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "labels": labels,
        "validation_method": "GroupShuffleSplit",
        "test_fleets": sorted(groups_test.unique().tolist()),
        "metrics": {
            key: float(value)
            for key, value in best_metrics.items()
            if key in [
                "f1_macro",
                "accuracy",
                "precision_macro",
                "recall_macro",
            ]
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    # Resumen legible para el README y la defensa.
    summary = [
        "# Resumen del entrenamiento",
        "",
        f"- Modelo seleccionado: {best_model_name}",
        f"- Variable objetivo: {target_column}",
        f"- Variables predictoras: {', '.join(feature_columns)}",
        "- Método de evaluación: GroupShuffleSplit por fleet_id",
        f"- Flotas de prueba: {', '.join(sorted(groups_test.unique()))}",
        f"- F1 macro: {best_metrics['f1_macro']:.4f}",
        f"- Accuracy: {best_metrics['accuracy']:.4f}",
        f"- Precision macro: {best_metrics['precision_macro']:.4f}",
        f"- Recall macro: {best_metrics['recall_macro']:.4f}",
        "",
        "## Comparación de modelos",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Reporte de clasificación",
        "",
        "```text",
        report,
        "```",
        "",
        "La matriz de confusión está disponible en "
        "`reports/confusion_matrix.csv`.",
    ]

    summary_path = REPORT_DIR / "training_summary.md"
    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\nCOMPARACIÓN DE MODELOS")
    print("------------------------------")
    print(comparison.to_string(index=False))

    print("\nMEJOR MODELO")
    print("------------------------------")
    print(best_model_name)
    print(f"F1 macro: {best_metrics['f1_macro']:.4f}")

    print("\nARCHIVOS GENERADOS")
    print("------------------------------")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Metadatos: {METADATA_PATH}")
    print(f"Comparación: {comparison_path}")
    print(f"Matriz de confusión: {matrix_path}")
    print(f"Reporte: {report_path}")
    print(f"Resumen: {summary_path}")


# ---------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------

def main():
    X, y, groups, target_column, feature_columns = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    ) = split_by_fleet(X, y, groups)

    models = build_models()

    comparison, trained_models = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_results(
        comparison,
        trained_models,
        X_test,
        y_test,
        groups_test,
        target_column,
        feature_columns,
    )

    print("\nENTRENAMIENTO COMPLETADO.")


if __name__ == "__main__":
    main()
