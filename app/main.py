
# ==========================================================
# ACI606 - CLOUD FLEET CHALLENGE
# FastAPI + PostgreSQL (Neon)
# ==========================================================

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError

from app.db import (
    SessionLocal,
    TelemetryRecord,
    init_db
)


# ==========================================================
# 1. CONFIGURACION DE LA APLICACION
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Inicializar las tablas de PostgreSQL
    init_db()

    print("Base de datos PostgreSQL inicializada.")

    yield

    print("Aplicacion finalizada.")


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.2.0",
    description="Cloud Fleet API with PostgreSQL persistence.",
    lifespan=lifespan
)


# ==========================================================
# 2. MODELOS DE DATOS
# ==========================================================

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):

    # Rechazar NaN e infinitos
    model_config = ConfigDict(
        allow_inf_nan=False
    )

    robot_id: str = Field(
        min_length=1,
        max_length=20
    )

    timestamp: float

    x: float
    y: float

    speed: float = Field(
        ge=0
    )

    battery: float = Field(
        ge=0,
        le=100
    )

    motor_temp: float

    vibration: float = Field(
        ge=0
    )

    motor_current: float = Field(
        ge=0
    )

    obstacle_distance: float = Field(
        ge=0
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
            "utc": datetime.now(
                timezone.utc
            ).isoformat()
        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        ) from error


# ==========================================================
# 5. INGESTA DE TELEMETRIA
# ==========================================================

@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:

    global REQUEST_COUNT

    try:

        with SessionLocal() as session:

            # Crear registro
            record = TelemetryRecord(
                **sample.model_dump()
            )

            # Guardar en PostgreSQL
            session.add(record)

            session.commit()

        REQUEST_COUNT += 1

        return {
            "accepted": True,
            "robot_id": sample.robot_id,
            "server_time": datetime.now(
                timezone.utc
            ).isoformat(),
            "note": "Data stored in PostgreSQL."
        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not store telemetry"
        ) from error


# ==========================================================
# 6. CONSULTAR ROBOTS
# ==========================================================

@app.get("/robots")
def list_robots() -> list[dict]:

    try:

        # Obtener el ultimo ID registrado de cada robot
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
                    TelemetryRecord.id == latest_ids.c.latest_id
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
            detail="Could not retrieve robots"
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
                    TelemetryRecord.id
                )
            ).all()

            if not records:

                raise HTTPException(
                    status_code=404,
                    detail="Robot not found"
                )

            return [
                record.to_dict()
                for record in records
            ]

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not retrieve robot history"
        ) from error


# ==========================================================
# 8. PREDICCION MACHINE LEARNING
# ==========================================================

@app.post("/predict")
def predict(_: PredictionRequest) -> dict:

    # Se implementara cuando tengamos el modelo entrenado
    raise HTTPException(
        status_code=501,
        detail="ML model integration is pending."
    )


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
            "prediction_service": "NOT_IMPLEMENTED"
        }

    except SQLAlchemyError as error:

        raise HTTPException(
            status_code=503,
            detail="Could not retrieve metrics"
        ) from error


# ==========================================================
# 10. DASHBOARD
# ==========================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse
)
def dashboard() -> str:

    dashboard_path = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "index.html"
    )

    if not dashboard_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found"
        )

    return dashboard_path.read_text(
        encoding="utf-8"
    )
