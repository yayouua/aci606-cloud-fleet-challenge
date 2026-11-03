
# ==========================================================
# ACI606 - CLOUD FLEET CHALLENGE
# FastAPI + PostgreSQL (Neon) + Machine Learning
# ==========================================================

from __future__ import annotations

import json
import os
import time

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
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
# 1. CONFIGURACION DE LA APLICACION
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


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Inicializar las tablas de PostgreSQL.
    init_db()

    print("Base de datos PostgreSQL inicializada.")

    # Cargar el modelo entrenado.
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"No se encontro el modelo entrenado: {MODEL_PATH}"
        )

    app.state.ml_model = joblib.load(MODEL_PATH)

    # Cargar metadatos del modelo.
    if MODEL_METADATA_PATH.is_file():
        metadata = json.loads(
            MODEL_METADATA_PATH.read_text(encoding="utf-8")
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
            "Los metadatos contienen variables no disponibles "
            "en el endpoint de prediccion."
        )

    app.state.model_metadata = metadata
    app.state.feature_columns = feature_columns

    app.state.model_version = metadata.get(
        "model_version",
        "v1",
    )

    print("Modelo ML cargado correctamente.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Variables: {feature_columns}")

    yield

    print("Aplicacion finalizada.")


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.3.0",
    description=(
        "Cloud Fleet API with PostgreSQL persistence "
        "and Machine Learning predictions."
    ),
    lifespan=lifespan,
)


# ==========================================================
# 2. MODELOS DE DATOS
# ==========================================================

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):

    # Rechazar NaN e infinitos.
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
# 3. CONTADORES BASICOS
# ==========================================================

REQUEST_COUNT = 0


# ==========================================================
# 4. HEALTH CHECK
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
# 5. INGESTA DE TELEMETRIA
# ==========================================================

@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:

    global REQUEST_COUNT

    try:

        with SessionLocal() as session:

            record = TelemetryRecord(
                **sample.model_dump()
            )

            session.add(record)
            session.commit()

        REQUEST_COUNT += 1

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
# 6. CONSULTAR ROBOTS
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
                    TelemetryRecord.id == latest_ids.c.latest_id,
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
# 7. CONSULTAR HISTORIAL COMPLETO
# ==========================================================

@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:

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
# 8. PREDICCION MACHINE LEARNING
# ==========================================================

@app.post("/predict")
def predict(data: PredictionRequest) -> dict:

    if not hasattr(app.state, "ml_model"):

        raise HTTPException(
            status_code=503,
            detail="ML model is not available.",
        )

    start_time = time.perf_counter()

    try:

        model = app.state.ml_model

        feature_columns = app.state.feature_columns

        # Obtener los datos de entrada.
        payload = data.model_dump()

        # Construir el DataFrame en el mismo orden
        # utilizado durante el entrenamiento.
        input_data = pd.DataFrame(
            [
                {
                    column: payload[column]
                    for column in feature_columns
                }
            ],
            columns=feature_columns,
        )

        # Obtener prediccion.
        prediction = model.predict(input_data)[0]

        # Obtener la probabilidad de cada clase.
        probabilities = model.predict_proba(input_data)[0]

        confidence = float(max(probabilities))

        # Medir tiempo de inferencia.
        inference_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "robot_id": data.robot_id,
            "predicted_status": str(prediction),
            "confidence": round(confidence, 4),
            "model_version": app.state.model_version,
            "inference_ms": round(inference_ms, 3),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail="Prediction could not be completed.",
        ) from error


# ==========================================================
# 9. METRICAS BASICAS
# ==========================================================

@app.get("/metrics")
def metrics() -> dict:

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
            "requests_received": REQUEST_COUNT,
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
# 10. DASHBOARD
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
