
# ==========================================================
# ACI606 - CLOUD FLEET CHALLENGE
# FastAPI + PostgreSQL (Neon) + Machine Learning
# Observabilidad y métricas operacionales
# ==========================================================

import json
import math
import os
import time

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError

from app.db import (
    SessionLocal,
    TelemetryRecord,
    init_db,
)


# ==========================================================
# 1. CONFIGURACION GENERAL
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(ROOT_DIR / "models" / "model.joblib"),
    )
)

MODEL_METADATA_PATH = (
    MODEL_PATH.parent / "model_metadata.json"
)

DEFAULT_FEATURES = [
    "x",
    "y",
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]


# ==========================================================
# 2. CONTADORES DE OBSERVABILIDAD
# ==========================================================

REQUEST_COUNT = 0

ERROR_COUNT = 0

TOTAL_LATENCY_MS = 0.0

APP_START_TIME = time.perf_counter()

# Se conservan las ultimas 10.000 mediciones.
LATENCY_SAMPLES = deque(maxlen=10000)


# ==========================================================
# 3. INICIALIZACION DE LA APLICACION
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global APP_START_TIME

    # Inicializar las tablas de PostgreSQL.
    init_db()

    print("Base de datos PostgreSQL inicializada.")

    # Comprobar que exista el modelo entrenado.
    if not MODEL_PATH.is_file():

        raise RuntimeError(
            f"No se encontro el modelo entrenado: {MODEL_PATH}"
        )

    # Cargar pipeline de Machine Learning.
    app.state.ml_model = joblib.load(MODEL_PATH)

    # Cargar metadatos.
    if MODEL_METADATA_PATH.is_file():

        metadata = json.loads(
            MODEL_METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        metadata = {}

    feature_columns = metadata.get(
        "feature_columns",
        DEFAULT_FEATURES,
    )

    if not feature_columns:

        raise RuntimeError(
            "El modelo no tiene variables predictoras configuradas."
        )

    if not set(feature_columns).issubset(DEFAULT_FEATURES):

        raise RuntimeError(
            "Los metadatos contienen variables que "
            "no estan disponibles en el endpoint."
        )

    app.state.model_metadata = metadata

    app.state.feature_columns = feature_columns

    app.state.model_version = metadata.get(
        "model_version",
        "v1",
    )

    # Comenzar a medir el uptime cuando la aplicacion
    # termina de inicializarse.
    APP_START_TIME = time.perf_counter()

    print("Modelo ML cargado correctamente.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Version: {app.state.model_version}")
    print(f"Variables: {feature_columns}")

    print("Servicio listo.")

    yield

    print("Aplicacion finalizada.")


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.4.0",
    description=(
        "Cloud Fleet API with PostgreSQL persistence, "
        "Machine Learning predictions and observability."
    ),
    lifespan=lifespan,
)


# ==========================================================
# 4. MIDDLEWARE DE OBSERVABILIDAD
# ==========================================================

@app.middleware("http")
async def observe_requests(
    request: Request,
    call_next,
):

    global REQUEST_COUNT
    global ERROR_COUNT
    global TOTAL_LATENCY_MS

    # Excluir /metrics para evitar que una consulta
    # modifique las mismas metricas que esta mostrando.
    if request.url.path == "/metrics":

        return await call_next(request)

    start_time = time.perf_counter()

    status_code = 500

    try:

        response = await call_next(request)

        status_code = response.status_code

        return response

    except Exception:

        status_code = 500

        raise

    finally:

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Contar solicitudes.
        REQUEST_COUNT += 1

        # Acumular latencia.
        TOTAL_LATENCY_MS += elapsed_ms

        # Guardar muestra de latencia.
        LATENCY_SAMPLES.append(elapsed_ms)

        # Contabilizar errores HTTP.
        if status_code >= 400:

            ERROR_COUNT += 1


# ==========================================================
# 5. CALCULO DE PERCENTILES DE LATENCIA
# ==========================================================

def calculate_percentile(
    values,
    percentile,
):

    if not values:

        return 0.0

    ordered = sorted(values)

    position = math.ceil(
        percentile * len(ordered)
    ) - 1

    position = max(
        0,
        min(position, len(ordered) - 1)
    )

    return round(
        ordered[position],
        3,
    )


# ==========================================================
# 6. MODELOS DE DATOS
# ==========================================================

Status = Literal[
    "NORMAL",
    "WARNING",
    "CRITICAL",
]


class Telemetry(BaseModel):

    model_config = ConfigDict(
        allow_inf_nan=False
    )

    robot_id: str = Field(
        min_length=1,
        max_length=20,
    )

    timestamp: float

    x: float

    y: float

    speed: float = Field(
        ge=0,
    )

    battery: float = Field(
        ge=0,
        le=100,
    )

    motor_temp: float

    vibration: float = Field(
        ge=0,
    )

    motor_current: float = Field(
        ge=0,
    )

    obstacle_distance: float = Field(
        ge=0,
    )


class PredictionRequest(Telemetry):
    pass


# ==========================================================
# 7. HEALTH CHECK
# ==========================================================

@app.get("/health")
def health() -> dict:

    try:

        with SessionLocal() as session:

            robot_count = session.scalar(
                select(
                    func.count(
                        func.distinct(
                            TelemetryRecord.robot_id
                        )
                    )
                )
            ) or 0

        return {

            "status": "ok",

            "service": "aci606-cloud-fleet",

            "version": app.version,

            "robots_in_database": robot_count,

            "persistence": "POSTGRESQL",

            "prediction_service": (
                "READY"
                if hasattr(app.state, "ml_model")
                else "NOT_READY"
            ),

            "utc": datetime.now(
                timezone.utc
            ).isoformat(),

        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from error


# ==========================================================
# 8. INGESTA DE TELEMETRIA
# ==========================================================

@app.post("/telemetry")
def ingest_telemetry(
    sample: Telemetry,
) -> dict:

    try:

        with SessionLocal() as session:

            record = TelemetryRecord(
                **sample.model_dump()
            )

            session.add(record)

            session.commit()

        return {

            "accepted": True,

            "robot_id": sample.robot_id,

            "server_time": datetime.now(
                timezone.utc
            ).isoformat(),

            "note": "Data stored in PostgreSQL.",

        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not store telemetry",
        ) from error


# ==========================================================
# 9. CONSULTAR ROBOTS
# ==========================================================

@app.get("/robots")
def list_robots() -> list[dict]:

    try:

        latest_ids = (
            select(
                func.max(
                    TelemetryRecord.id
                ).label("latest_id")
            )
            .group_by(
                TelemetryRecord.robot_id
            )
            .subquery()
        )

        with SessionLocal() as session:

            records = session.scalars(

                select(TelemetryRecord)
                .join(
                    latest_ids,
                    TelemetryRecord.id
                    == latest_ids.c.latest_id,
                )
                .order_by(
                    TelemetryRecord.robot_id
                )

            ).all()

            return [
                record.to_dict()
                for record in records
            ]

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not retrieve robots",
        ) from error


# ==========================================================
# 10. CONSULTAR HISTORIAL COMPLETO
# ==========================================================

@app.get("/robots/{robot_id}/history")
def robot_history(
    robot_id: str,
) -> list[dict]:

    try:

        with SessionLocal() as session:

            records = session.scalars(

                select(TelemetryRecord)
                .where(
                    TelemetryRecord.robot_id == robot_id
                )
                .order_by(
                    TelemetryRecord.timestamp,
                    TelemetryRecord.id,
                )

            ).all()

            if not records:

                raise HTTPException(
                    status_code=404,
                    detail="Robot not found",
                )

            return [
                record.to_dict()
                for record in records
            ]

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not retrieve robot history",
        ) from error


# ==========================================================
# 11. PREDICCION MACHINE LEARNING
# ==========================================================

@app.post("/predict")
def predict(
    data: PredictionRequest,
) -> dict:

    if not hasattr(app.state, "ml_model"):

        raise HTTPException(
            status_code=503,
            detail="ML model is not available.",
        )

    start_time = time.perf_counter()

    try:

        model = app.state.ml_model

        feature_columns = app.state.feature_columns

        # Convertir entrada a diccionario.
        payload = data.model_dump()

        # Construir DataFrame respetando
        # el orden de entrenamiento.
        input_data = pd.DataFrame(
            [
                {
                    column: payload[column]
                    for column in feature_columns
                }
            ],
            columns=feature_columns,
        )

        # Ejecutar prediccion.
        prediction = model.predict(
            input_data
        )[0]

        # Obtener probabilidades.
        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = float(
            max(probabilities)
        )

        # Medir latencia de inferencia.
        inference_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {

            "robot_id": data.robot_id,

            "predicted_status": str(prediction),

            "confidence": round(
                confidence,
                4,
            ),

            "model_version": app.state.model_version,

            "inference_ms": round(
                inference_ms,
                3,
            ),

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail="Prediction could not be completed.",
        ) from error


# ==========================================================
# 12. METRICAS OPERACIONALES
# ==========================================================

@app.get("/metrics")
def metrics() -> dict:

    try:

        # Consultar robots registrados en PostgreSQL.
        with SessionLocal() as session:

            robot_count = session.scalar(
                select(
                    func.count(
                        func.distinct(
                            TelemetryRecord.robot_id
                        )
                    )
                )
            ) or 0

        # Tiempo de funcionamiento.
        uptime_seconds = (
            time.perf_counter() - APP_START_TIME
        )

        # Copiar las muestras de latencia.
        latency_samples = list(
            LATENCY_SAMPLES
        )

        # Calcular latencia promedio.
        average_latency = (

            TOTAL_LATENCY_MS / REQUEST_COUNT

            if REQUEST_COUNT > 0

            else 0.0

        )

        # Calcular tasa de errores.
        error_rate = (

            ERROR_COUNT / REQUEST_COUNT * 100

            if REQUEST_COUNT > 0

            else 0.0

        )

        # Calcular percentiles.
        latency_p50 = calculate_percentile(
            latency_samples,
            0.50,
        )

        latency_p95 = calculate_percentile(
            latency_samples,
            0.95,
        )

        return {

            "requests_received": REQUEST_COUNT,

            "errors_total": ERROR_COUNT,

            "error_rate_percent": round(
                error_rate,
                2,
            ),

            "latency_avg_ms": round(
                average_latency,
                3,
            ),

            "latency_p50_ms": latency_p50,

            "latency_p95_ms": latency_p95,

            "latency_samples": len(
                latency_samples
            ),

            "uptime_seconds": round(
                uptime_seconds,
                2,
            ),

            "robots_in_database": robot_count,

            "persistence": "POSTGRESQL",

            "prediction_service": (
                "READY"
                if hasattr(app.state, "ml_model")
                else "NOT_READY"
            ),

        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not retrieve metrics",
        ) from error


# ==========================================================
# 13. DASHBOARD
# ==========================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
)
def dashboard() -> str:

    dashboard_path = (
        ROOT_DIR
        / "dashboard"
        / "index.html"
    )

    if not dashboard_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found",
        )

    return dashboard_path.read_text(
        encoding="utf-8"
    )


# ==========================================================
# FIN DE LA APLICACION
# ==========================================================
