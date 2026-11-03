# Cloud Fleet Challenge

**ACI606 – Cloud Computing**  
**Ejercicio 4: Del prototipo local a un servicio desplegado en una plataforma PaaS**

**Repositorio:** [yayouua/aci606-cloud-fleet-challenge](https://github.com/yayouua/aci606-cloud-fleet-challenge)

**Servicio desplegado:** [Cloud Fleet Challenge – Render](https://aci606-cloud-fleet-challenge-h8wb.onrender.com/)

---

## 1. Descripción del proyecto

Cloud Fleet Challenge es un sistema de monitoreo, almacenamiento persistente y predicción del estado operacional de una flota de robots mediante Machine Learning.

El proyecto transforma una aplicación inicialmente ejecutada de manera local en un servicio accesible desde internet, utilizando FastAPI, PostgreSQL y una plataforma de despliegue PaaS.

El sistema permite:

- Recibir información de telemetría de robots.
- Almacenar los registros de forma persistente en PostgreSQL.
- Consultar los robots registrados y su última telemetría.
- Consultar el historial completo de cada robot.
- Predecir el estado operacional mediante Machine Learning.
- Visualizar información mediante un dashboard web.
- Supervisar el funcionamiento del servicio mediante métricas operacionales.
- Ejecutar la aplicación desde una URL pública HTTPS.

Las clases utilizadas para las predicciones son:

- `NORMAL`
- `WARNING`
- `CRITICAL`

## 2. URL pública del servicio

La aplicación está desplegada en Render y puede consultarse públicamente:

**URL principal:**

https://aci606-cloud-fleet-challenge-h8wb.onrender.com/

**Recursos disponibles:**

| Recurso | URL | Función |
|---|---|---|
| API | `/` | Servicio principal |
| Health | `/health` | Estado del servicio |
| Documentación | `/docs` | Documentación interactiva de FastAPI |
| Dashboard | `/dashboard` | Panel de monitoreo |
| Robots | `/robots` | Listado de robots registrados |
| Métricas | `/metrics` | Métricas operacionales |

## 3. Arquitectura del sistema

La arquitectura implementada utiliza los siguientes componentes:

1. **Simulador:** genera datos de telemetría que son enviados a la API.
2. **FastAPI:** recibe solicitudes, procesa información y ejecuta predicciones.
3. **PostgreSQL con Neon:** almacena persistentemente la telemetría de los robots.
4. **Machine Learning:** utiliza un pipeline entrenado mediante Scikit-learn.
5. **Joblib:** permite guardar y cargar el modelo entrenado.
6. **Dashboard:** presenta información operacional mediante HTML, CSS y JavaScript.
7. **Render:** aloja y ejecuta el servicio web mediante una plataforma PaaS.

### Diagrama de arquitectura

```text
               SIMULADOR
                   |
                   | Telemetría
                   v
              FASTAPI
                   |
        +----------+----------+
        |          |          |
        v          v          v
   PostgreSQL   Modelo ML   Dashboard
     (Neon)    (Joblib)    (HTML/CSS/JS)
        |          |          |
        +----------+----------+
                   |
              OBSERVABILIDAD
                   |
                   v
                RENDER
             Servicio HTTPS
```

La base de datos Neon mantiene los registros de telemetría, mientras que Render ejecuta la aplicación y carga el modelo de Machine Learning.

Esta separación permite conservar los datos aunque el servicio web sea reiniciado.

## 4. Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje de programación |
| FastAPI | Desarrollo de la API REST |
| Uvicorn | Servidor de aplicaciones |
| PostgreSQL | Base de datos relacional |
| Neon | PostgreSQL gestionado en la nube |
| SQLAlchemy | Gestión de conexiones y consultas |
| Psycopg | Conector de PostgreSQL |
| Pandas | Manipulación de datos |
| NumPy | Operaciones numéricas |
| Scikit-learn | Entrenamiento y evaluación de modelos |
| Joblib | Serialización del modelo |
| HTML | Estructura del dashboard |
| CSS | Estilos de la interfaz |
| JavaScript | Interacción con la API |
| Render | Plataforma de despliegue PaaS |
| Git y GitHub | Control de versiones |

## 5. Estructura del repositorio

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

Los directorios `models/`, `data/` y `reports/` contienen los artefactos del entrenamiento, el dataset y los resultados del análisis.

## 6. Instalación y ejecución local

### 6.1. Requisitos

Para ejecutar el proyecto localmente se necesita:

- Python.
- Git.
- Una cuenta y base de datos PostgreSQL en Neon.
- Las dependencias indicadas en `requirements.txt`.

### 6.2. Clonar el repositorio

```bash
git clone https://github.com/yayouua/aci606-cloud-fleet-challenge.git

cd aci606-cloud-fleet-challenge
```

### 6.3. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 6.4. Configurar variables de entorno

Crear un archivo `.env` en la raíz del repositorio.

Configurar las siguientes variables:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require

MODEL_PATH=models/model.joblib
```

Los valores de conexión deben reemplazarse por las credenciales reales de la base de datos Neon.

El archivo `.env` contiene información privada y no debe subirse a GitHub.

El archivo `.env.example` proporciona una plantilla sin credenciales reales.

### 6.5. Inicializar la base de datos

```bash
python -c "from app.db import init_db; init_db()"
```

La aplicación también inicializa la base de datos durante el arranque.

### 6.6. Ejecutar FastAPI

```bash
python -m uvicorn app.main:app --reload
```

La aplicación estará disponible en:

```text
http://127.0.0.1:8000
```

Documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

Dashboard:

```text
http://127.0.0.1:8000/dashboard
```

## 7. Generación del dataset y análisis exploratorio

El proyecto utiliza un dataset de telemetría de robots para entrenar los clasificadores.

El dataset contiene variables relacionadas con:

- Posición X.
- Posición Y.
- Velocidad.
- Nivel de batería.
- Temperatura del motor.
- Vibración.
- Corriente del motor.
- Distancia a obstáculos.
- Identificador de flota (`fleet_id`).
- Estado operacional (`status`).

El análisis exploratorio permite estudiar la distribución de las clases, la distribución de las flotas, los valores faltantes y las estadísticas descriptivas.

Los resultados del análisis se almacenan en el directorio `reports/`.

## 8. Entrenamiento y evaluación de Machine Learning

Se compararon dos clasificadores de Scikit-learn:

1. Random Forest.
2. Logistic Regression.

La evaluación se realizó utilizando `GroupShuffleSplit`, agrupando los datos mediante `fleet_id`.

Este procedimiento permite separar flotas completas entre entrenamiento y prueba y evaluar el modelo sobre flotas que no participaron en el entrenamiento.

### 8.1. Dataset utilizado

| Indicador | Resultado |
|---|---:|
| Registros válidos | 3.000 |
| Flotas | 20 |
| Clases | 3 |
| Registros de entrenamiento | 2.250 |
| Registros de prueba | 750 |
| Flotas de entrenamiento | 15 |
| Flotas de prueba | 5 |
| Flotas compartidas | 0 |

### 8.2. Comparación de clasificadores

Resultados obtenidos durante el entrenamiento:

| Modelo | F1 macro | Accuracy |
|---|---:|---:|
| Random Forest | 0.9350 | 0.9320 |
| Logistic Regression | **0.9368** | **0.9333** |

El modelo seleccionado fue Logistic Regression, porque obtuvo el mayor F1 macro durante la evaluación.

El F1 macro permite evaluar el rendimiento considerando por igual las distintas clases, incluyendo `CRITICAL`, que puede tener una representación menor en el dataset.

### 8.3. Métrica principal

La métrica principal utilizada fue F1 macro:

\[
F1_{\text{macro}} =
\frac{1}{K}\sum_{k=1}^{K}F1_k
\]

Donde cada clase recibe el mismo peso en el promedio.

### 8.4. Artefactos generados

Los resultados se guardan en:

```text
models/
├── model.joblib
└── model_metadata.json

reports/
├── model_comparison.csv
├── confusion_matrix.csv
├── classification_report.txt
└── training_summary.md
```

El pipeline entrenado se almacena en `models/model.joblib` y es cargado por FastAPI durante el inicio del servicio.

### 8.5. Ejecutar nuevamente el entrenamiento

```bash
python ml/train_model.py
```

## 9. Endpoints de la API

El servicio implementa las siguientes rutas:

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Verifica el estado del servicio |
| POST | `/telemetry` | Registra una muestra de telemetría |
| GET | `/robots` | Consulta los robots registrados |
| GET | `/robots/{id}/history` | Obtiene el historial completo de un robot |
| POST | `/predict` | Predice el estado operacional |
| GET | `/metrics` | Consulta las métricas operacionales |
| GET | `/dashboard` | Presenta el panel de monitoreo |

### 9.1. Ejemplo de telemetría

Solicitud POST:

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

### 9.2. Ejemplo de respuesta de `/predict`

```json
{
  "robot_id": "R03",
  "predicted_status": "NORMAL",
  "confidence": 1.0,
  "model_version": "v1",
  "inference_ms": 34.883
}
```

Los valores de confianza y latencia dependen de los datos enviados y del procesamiento de cada solicitud.

### 9.3. Validación de datos

La API utiliza modelos Pydantic para validar las entradas.

Entre las validaciones implementadas se encuentran:

- Identificador de robot obligatorio.
- Rechazo de valores NaN e infinitos.
- Batería entre 0 y 100.
- Velocidad no negativa.
- Vibración no negativa.
- Corriente del motor no negativa.
- Distancia a obstáculos no negativa.

Las solicitudes con datos inválidos son rechazadas mediante respuestas HTTP 4xx, sin convertir los errores de validación en errores internos 5xx.

## 10. Persistencia de datos con Neon PostgreSQL

El servicio utiliza PostgreSQL gestionado mediante Neon.

La telemetría recibida se almacena en la base de datos utilizando SQLAlchemy y Psycopg.

El historial se consulta mediante:

```text
GET /robots/{id}/history
```

La persistencia permite conservar los registros independientemente de los reinicios del servicio desplegado en Render.

### Prueba de persistencia realizada

Se utilizó el robot de prueba `RFINAL18`.

Resultados:

| Prueba | Resultado |
|---|---|
| Envío de telemetría | Correcto |
| Cantidad de registros enviados | 120 |
| Consulta de historial | Correcta |
| Registros recuperados | 120 |
| Persistencia después del reinicio | Correcta |

Esta prueba permite comprobar que el historial se conserva en PostgreSQL y no depende únicamente de la memoria de la aplicación.

## 11. Dashboard de operación

El sistema incorpora un dashboard desarrollado con HTML, CSS y JavaScript.

El panel presenta:

- Cantidad de robots registrados.
- Nivel de batería.
- Velocidad.
- Temperatura del motor.
- Posición del robot.
- Vibración.
- Corriente del motor.
- Distancia a obstáculos.
- Predicción del estado operacional.
- Confianza del modelo.
- Tiempo de inferencia.
- Historial de telemetría.

El dashboard consulta los datos directamente a través de los endpoints de FastAPI y se actualiza automáticamente cada 15 segundos.

El historial completo permanece disponible mediante la API, mientras que el panel presenta una vista de los registros más recientes.

Dashboard público:

https://aci606-cloud-fleet-challenge-h8wb.onrender.com/dashboard

## 12. Observabilidad y métricas operacionales

La aplicación incorpora un sistema de observabilidad mediante middleware de FastAPI.

El endpoint de métricas es:

```text
GET /metrics
```

Se registran los siguientes indicadores:

| Métrica | Descripción |
|---|---|
| `requests_received` | Solicitudes recibidas |
| `errors_total` | Total de errores HTTP |
| `error_rate_percent` | Porcentaje de errores |
| `latency_avg_ms` | Latencia promedio |
| `latency_p50_ms` | Percentil 50 de latencia |
| `latency_p95_ms` | Percentil 95 de latencia |
| `latency_samples` | Cantidad de muestras de latencia conservadas |
| `uptime_seconds` | Tiempo de funcionamiento |
| `robots_in_database` | Cantidad de robots registrados |
| `prediction_service` | Estado del servicio de predicción |

Las muestras de latencia se conservan en memoria hasta un máximo de 10.000 observaciones.

Los contadores y las métricas de ejecución se reinician al reiniciar la aplicación, mientras que los registros de telemetría permanecen almacenados en Neon.

## 13. Pruebas de validación del servicio desplegado

Las pruebas finales se ejecutaron contra la URL pública de Render.

### 13.1. Verificación de endpoints

| Prueba | Respuesta HTTP | Resultado |
|---|---:|---|
| GET `/health` | 200 | Correcto |
| GET `/dashboard` | 200 | Correcto |
| GET `/robots` | 200 | Correcto |
| GET `/metrics` | 200 | Correcto |
| GET `/robots/RFINAL18/history` | 200 | Correcto |
| POST `/predict` válido | 200 | Correcto |
| POST `/predict` con batería inválida | 422 | Correcto |

### 13.2. Prueba de historial

Se enviaron 120 registros al robot `RFINAL18`.

Se verificó que:

- Los registros fueron aceptados.
- El historial pudo recuperarse mediante la API.
- Se recuperaron los 120 registros.
- Los datos permanecieron disponibles después del reinicio del servicio.

**Resultado: prueba de persistencia superada.**

### 13.3. Prueba de concurrencia

Se realizaron 40 solicitudes POST `/predict` utilizando una concurrencia máxima de 8 solicitudes.

La prueba permitió evaluar:

- Respuestas HTTP.
- Solicitudes exitosas.
- Solicitudes fallidas.
- Latencia individual.
- Percentiles P50 y P95.

Resultado informado durante la validación: solicitudes concurrentes procesadas correctamente, sin fallos observados.

### 13.4. Prueba de validación de datos inválidos

Se envió una solicitud con una batería de valor 150.

La API respondió:

```text
HTTP 422 Unprocessable Content
```

Esto demuestra que los datos fuera de rango son rechazados mediante validación HTTP 4xx.

## 14. Despliegue en Render

La aplicación está desplegada mediante Render Web Service.

### 14.1. Configuración de construcción

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 14.2. Variables de entorno

El servicio utiliza las siguientes variables:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require

MODEL_PATH=models/model.joblib
```

Estas variables se configuran desde el panel de Render.

No se almacenan credenciales reales en el repositorio.

### 14.3. Componentes desplegados

| Componente | Plataforma |
|---|---|
| API y aplicación FastAPI | Render |
| Base de datos | Neon PostgreSQL |
| Modelo de Machine Learning | Archivo `models/model.joblib` |
| Dashboard | FastAPI y HTML |
| Repositorio | GitHub |

### 14.4. Consideraciones operacionales

El servicio utiliza una instancia de Render y una base de datos PostgreSQL gestionada.

En planes gratuitos, el servicio web puede suspenderse después de períodos de inactividad y el primer acceso posterior puede tardar más debido al proceso de reactivación.

El almacenamiento persistente de telemetría se mantiene en Neon para evitar depender del disco efímero de la instancia de Render.

## 15. Control de versiones y trazabilidad Git

El desarrollo se organizó mediante Git utilizando una rama de funcionalidad:

```text
feature/cloud-deployment
```

Los cambios se registraron mediante commits independientes para mantener la trazabilidad de las funcionalidades implementadas.

Entre los cambios realizados se incluyen:

- Configuración de PostgreSQL y modelo de telemetría.
- Entrenamiento y comparación de clasificadores.
- Integración del endpoint de predicción.
- Desarrollo del dashboard.
- Implementación de observabilidad.
- Documentación y despliegue.

El flujo de trabajo contempla integrar la rama de funcionalidad en `main` mediante un Pull Request dirigido al repositorio propio:

https://github.com/yayouua/aci606-cloud-fleet-challenge

## 16. Seguridad y configuración

Para proteger las credenciales y la información de conexión:

- Las credenciales de Neon se almacenan en variables de entorno.
- El archivo `.env` no debe subirse a GitHub.
- `.env.example` contiene solamente valores de ejemplo.
- Las variables sensibles se configuran directamente en Render.
- No se deben publicar contraseñas, claves privadas ni cadenas de conexión reales.

## 17. Conclusiones

Cloud Fleet Challenge permite transformar un prototipo local de monitoreo de robots en un servicio web desplegado en una plataforma PaaS.

La solución integra almacenamiento persistente, procesamiento de telemetría, modelos de clasificación, inferencia mediante una API REST, visualización operacional y métricas de observabilidad.

La evaluación de Machine Learning permitió comparar dos clasificadores utilizando separación por flotas, seleccionando Logistic Regression por su mayor F1 macro en la evaluación realizada.

La integración de FastAPI, Neon PostgreSQL y Render permite mantener el servicio accesible públicamente y conservar los registros de telemetría independientemente de los reinicios de la aplicación.

Las pruebas realizadas demostraron el funcionamiento de los endpoints, la validación de entradas, la recuperación de historiales superiores a 100 registros y la atención de solicitudes concurrentes.

---

**Asignatura:** ACI606 – Cloud Computing  
**Proyecto:** Ejercicio 4 – Cloud Fleet Challenge  
**Repositorio:** https://github.com/yayouua/aci606-cloud-fleet-challenge  
**Servicio público:** https://aci606-cloud-fleet-challenge-h8wb.onrender.com/