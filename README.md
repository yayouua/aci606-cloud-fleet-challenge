
# Cloud Fleet Challenge

**ACI606 - Cloud Computing**

Sistema de monitoreo, almacenamiento persistente y predicción del estado de robots mediante Machine Learning, desarrollado con FastAPI y PostgreSQL.

## 1. Descripción del proyecto

Cloud Fleet Challenge transforma un prototipo local de monitoreo de robots en un servicio preparado para ejecutarse en una plataforma PaaS.

El sistema permite:

- Recibir telemetría de robots.
- Almacenar los registros de manera persistente en PostgreSQL.
- Consultar robots y su historial completo.
- Predecir el estado de los robots mediante Machine Learning.
- Visualizar la información mediante un dashboard web.
- Supervisar el funcionamiento mediante métricas operacionales.

Las clases de predicción son:

- NORMAL
- WARNING
- CRITICAL

## 2. Arquitectura

La arquitectura del proyecto está compuesta por las siguientes partes:

1. **Simulador:** genera datos de telemetría y solicitudes.
2. **FastAPI:** recibe solicitudes, entrega información y ejecuta predicciones.
3. **PostgreSQL (Neon):** almacena de manera persistente los registros de telemetría.
4. **Machine Learning:** utiliza un pipeline entrenado y almacenado en `models/model.joblib`.
5. **Dashboard:** permite visualizar los robots, su telemetría, historial y predicciones.
6. **PaaS:** Render se utiliza como plataforma prevista para publicar el servicio.

Flujo general:

```text
Simulador
    |
    v
FastAPI
    |
    +---- PostgreSQL (Neon)
    |
    +---- Modelo ML (model.joblib)
    |
    +---- Dashboard
    |
    +---- Métricas operacionales
    |
    v
Servicio desplegado en Render
```

## 3. Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje de programación |
| FastAPI | API REST |
| Uvicorn | Servidor de aplicaciones |
| PostgreSQL | Base de datos |
| Neon | PostgreSQL gestionado |
| SQLAlchemy | Acceso a la base de datos |
| Psycopg | Conector PostgreSQL |
| Pandas | Procesamiento de datos |
| Scikit-learn | Entrenamiento y evaluación de modelos |
| Joblib | Serialización del modelo |
| HTML, CSS y JavaScript | Dashboard |
| Render | Plataforma prevista para el despliegue |

## 4. Estructura del proyecto

```text
aci606-cloud-fleet-challenge/
│
├── app/
│   ├── main.py
│   └── db.py
│
├── dashboard/
│   └── index.html
│
├── ml/
│   ├── generate_dataset.py
│   ├── eda.py
│   └── train_model.py
│
├── data/
│   └── training.csv
│
├── models/
│   ├── model.joblib
│   └── model_metadata.json
│
├── reports/
│   ├── model_comparison.csv
│   ├── confusion_matrix.csv
│   ├── classification_report.txt
│   └── training_summary.md
│
├── simulator/
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Configuración local

### Requisitos

- Python instalado.
- Git.
- Una base de datos PostgreSQL.
- Variables de conexión configuradas.

### Clonar el repositorio

```bash
git clone https://github.com/yayouua/aci606-cloud-fleet-challenge.git

cd aci606-cloud-fleet-challenge
```

### Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto.

Configurar:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
MODEL_PATH=models/model.joblib
```

Reemplazar los valores de conexión por los datos de la propia base Neon.

No publicar el archivo `.env` ni compartir sus credenciales.

### Inicializar la base de datos

```bash
python -c "from app.db import init_db; init_db()"
```

### Ejecutar la aplicación

```bash
python -m uvicorn app.main:app --reload
```

La API estará disponible localmente en:

```text
http://127.0.0.1:8000
```

La documentación interactiva estará disponible en:

```text
http://127.0.0.1:8000/docs
```

El dashboard estará disponible en:

```text
http://127.0.0.1:8000/dashboard
```

## 6. Entrenamiento de Machine Learning

El entrenamiento utiliza un dataset que contiene registros de distintas flotas de robots.

Se comparan dos clasificadores:

- Random Forest.
- Logistic Regression.

La separación entre entrenamiento y prueba se realiza mediante `GroupShuffleSplit`, utilizando `fleet_id` para evaluar el comportamiento del modelo frente a flotas no vistas.

### Ejecutar entrenamiento

```bash
python ml/train_model.py
```

### Resultados de la comparación

En la ejecución registrada se obtuvieron los siguientes resultados:

| Modelo | F1 macro | Accuracy |
|---|---:|---:|
| Random Forest | 0.9350 | 0.9320 |
| Logistic Regression | 0.9368 | 0.9333 |

El modelo seleccionado fue Logistic Regression, por presentar el mayor F1 macro en esa evaluación.

La evaluación utilizó 2.250 registros para entrenamiento y 750 para prueba, con 15 flotas de entrenamiento y 5 flotas de prueba sin intersección.

### Artefactos generados

Los resultados del entrenamiento se almacenan en:

```text
models/model.joblib
models/model_metadata.json
reports/model_comparison.csv
reports/confusion_matrix.csv
reports/classification_report.txt
reports/training_summary.md
```

El modelo serializado se utiliza posteriormente en el endpoint `/predict`.

## 7. Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/telemetry` | Registrar telemetría |
| GET | `/robots` | Consultar robots |
| GET | `/robots/{id}/history` | Consultar historial |
| POST | `/predict` | Predecir estado del robot |
| GET | `/metrics` | Consultar métricas |
| GET | `/dashboard` | Panel de operación |

### Ejemplo de telemetría y predicción

Solicitud:

```json
{
  "robot_id": "R03",
  "timestamp": 1793650000,
  "x": 1.2,
  "y": 3.4,
  "speed": 0.8,
  "battery": 85,
  "motor_temp": 42,
  "vibration": 0.2,
  "motor_current": 1.5,
  "obstacle_distance": 2.0
}
```

Respuesta esperada de `/predict`:

```json
{
  "robot_id": "R03",
  "predicted_status": "NORMAL",
  "confidence": 0.95,
  "model_version": "v1",
  "inference_ms": 12.4
}
```

Los valores de confianza y latencia del ejemplo son ilustrativos.

Los datos inválidos, como batería fuera del intervalo permitido o campos faltantes, deben ser rechazados con respuestas HTTP 4xx.

## 8. Persistencia de datos

La aplicación utiliza PostgreSQL gestionado mediante Neon.

Los registros se guardan en la base de datos y pueden recuperarse mediante:

```text
GET /robots
GET /robots/{id}/history
```

El historial no está limitado a los últimos 100 registros en memoria.

La persistencia debe verificarse enviando telemetría, reiniciando el servicio y consultando nuevamente el historial del robot.

## 9. Observabilidad

El endpoint:

```text
GET /metrics
```

permite consultar:

- Solicitudes recibidas.
- Total de errores.
- Tasa de errores.
- Latencia promedio.
- Latencia P50.
- Latencia P95.
- Tiempo de funcionamiento.
- Cantidad de robots registrados.
- Estado del servicio de predicción.

Las métricas de latencia y uptime se mantienen en memoria durante la ejecución del servicio.

## 10. Despliegue en Render

El despliegue previsto utiliza Render como Web Service y Neon como PostgreSQL gestionado.

### Comando de instalación

```bash
pip install -r requirements.txt
```

### Comando de inicio

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Variables de entorno

Configurar en Render:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
MODEL_PATH=models/model.joblib
```

El archivo `.env` local no se utiliza como sustituto de la configuración de variables de entorno de Render.

### URL pública

**Estado:** pendiente de completar después del despliegue.

URL del servicio:

```text
PENDIENTE: agregar URL pública de Render
```

### Ubicación de los datos

Los datos persistentes se almacenan en PostgreSQL mediante Neon.

## 11. Pruebas y validación

El sistema debe verificarse mediante las siguientes pruebas:

- Disponibilidad de `/health`.
- Ingesta de telemetría.
- Consulta de robots.
- Consulta del historial completo.
- Predicciones con `/predict`.
- Rechazo de datos inválidos mediante HTTP 4xx.
- Consulta de métricas.
- Persistencia después de reiniciar el servicio.
- Solicitudes concurrentes y medición de latencia.
- Evaluación con flotas no utilizadas durante el entrenamiento.

La prueba de despliegue debe ejecutarse contra la URL pública del servicio.

## 12. Control de versiones

El desarrollo se realiza utilizando Git y una rama de funcionalidad:

```text
feature/cloud-deployment
```

Los cambios se registran mediante commits independientes para:

- Conexión con PostgreSQL.
- Integración del modelo de Machine Learning.
- Endpoint de predicción.
- Dashboard.
- Observabilidad.
- Documentación.
- Despliegue.

La rama de funcionalidad debe integrarse finalmente en `main` mediante un Pull Request dirigido al repositorio del propio grupo.

## 13. Seguridad

- No almacenar credenciales directamente en el código.
- Mantener `.env` fuera del repositorio.
- Utilizar `.env.example` como plantilla.
- Configurar secretos en las variables de entorno de la plataforma.
- No publicar contraseñas de PostgreSQL ni claves privadas.

## 14. Autores

Proyecto académico desarrollado para:

**ACI606 - Cloud Computing**

Ejercicio 4: Cloud Fleet Challenge.
