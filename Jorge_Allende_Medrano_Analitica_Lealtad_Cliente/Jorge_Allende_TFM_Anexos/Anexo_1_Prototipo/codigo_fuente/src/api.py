"""API HTTP del prototipo de scoring — TFM Fase 8 (Productivización).

Expone tfm_pipeline.customer_features() + tfm_pipeline.score_customer() —la MISMA
lógica que usa el notebook notebooks/05_Conclusions_Productization.ipynb— como un
servicio FastAPI. No hay ninguna regla de negocio reimplementada aquí: este archivo
sólo orquesta la carga de artefactos y el I/O HTTP (JSON in, JSON out), tal como
enseña la guía "Productivizar Modelos en Python" (Carlos Ortega, UCM) — nivel 1 de
los "tres niveles de escalabilidad": exponer el modelo a un API. Contenedor y
orquestación quedan fuera de alcance.

Arranque local (entorno ml_env, el mismo que produjo los .pkl).
Ejecutar SIEMPRE desde la raíz del repo (api.py resuelve data/ y models/ como rutas
relativas a la raíz, y se importa como paquete src.api):
    conda run -n ml_env uvicorn src.api:app --reload --port 8000

Prototipo visual: http://127.0.0.1:8000/ (static/index.html — llama a este mismo API
por fetch(), mismo origen, sin servicio ni despliegue separado).

Documentación interactiva autogenerada (Swagger UI): http://127.0.0.1:8000/docs
Útil para el cuerpo del TFM: una captura de / (o de /docs) vale más que un fragmento
de código para el >=50% de la memoria orientado a audiencia no técnica.

Prueba rápida:
    curl -X POST http://127.0.0.1:8000/score \
         -H "Content-Type: application/json" \
         -d '{"customer_id": 16062}'
"""

from __future__ import annotations

import pickle
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import tfm_pipeline as tfm

app = FastAPI(
    title="TFM · Scoring de lealtad del cliente",
    description=(
        "Riesgo de churn a 90 días, CLV esperado y decisión de retención, "
        "a partir de transacciones crudas de un cliente. Online Retail II — UCM."
    ),
    version="1.0.0",
)

# --- Estado cargado UNA VEZ al arrancar el proceso, no en cada request ----------
# Los mismos artefactos que lee el notebook 05: ningún parámetro (umbral, corte de
# valor, factor de smearing, nombre del modelo ganador...) está escrito a mano aquí.
with open("models/models_fase3.pkl", "rb") as f:
    FASE3 = pickle.load(f)
with open("models/models_clv.pkl", "rb") as f:
    FASECLV = pickle.load(f)

TRANSACCIONES = pd.read_csv("data/processed/online_retail_II_cleaned.csv", parse_dates=["InvoiceDate"])
TRANSACCIONES["Customer ID"] = TRANSACCIONES["Customer ID"].astype(int)

NOMBRE_CHURN = FASE3["model_names"][FASE3["best_model_key"]]
NOMBRE_CLV = FASECLV["model_names"][FASECLV["best_model_key"]]


class ScoreRequest(BaseModel):
    customer_id: int = Field(
        ..., description="Customer ID del dataset Online Retail II.", examples=[16062]
    )
    as_of_date: Optional[str] = Field(
        None,
        description=(
            "Fecha de referencia ISO (YYYY-MM-DD). Por defecto, el día siguiente a la "
            "última transacción disponible en el histórico cargado (simula 'hoy')."
        ),
    )


class ScoreResponse(BaseModel):
    customer_id: int
    as_of_date: str
    riesgo_churn_90d: float
    clv_si_vuelve: float
    valor_esperado: float
    valor_en_riesgo: float
    decision: str
    accion: str


@app.get("/", include_in_schema=False)
def ui():
    """Sirve el prototipo visual (static/index.html) — mismo servicio, mismo puerto.

    HTML/CSS/JS estático sin dependencias nuevas: llama a /score y /health por
    fetch(), mismo origen. No hay lógica de negocio aquí, ni una segunda copia
    de la API.
    """
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    """Metadatos del servicio: qué modelo está sirviendo y con qué parámetros."""
    return {
        "status": "ok",
        "modelo_churn": NOMBRE_CHURN,
        "modelo_clv": NOMBRE_CLV,
        "umbral_churn": FASE3["threshold"],
        "corte_valor_clv": FASECLV["clv_median"],
        "lookback_months": tfm.LOOKBACK_MONTHS,
        "horizon_days": tfm.HORIZON_DAYS,
        "transacciones_cargadas": int(len(TRANSACCIONES)),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Cliente -> riesgo de churn, CLV esperado y decisión de retención."""
    as_of = (
        pd.to_datetime(req.as_of_date)
        if req.as_of_date
        else TRANSACCIONES["InvoiceDate"].max() + pd.Timedelta(days=1)
    )

    resultado = tfm.score_customer_from_transactions(
        TRANSACCIONES, req.customer_id, as_of, FASE3, FASECLV
    )

    if resultado is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Cliente {req.customer_id} sin compras en los {tfm.LOOKBACK_MONTHS} "
                f"meses previos a {as_of.date()}: no es scoreable con este modelo "
                f"(mismo criterio de población que el entrenamiento)."
            ),
        )

    resultado.pop("_features")
    return {"customer_id": req.customer_id, "as_of_date": str(as_of.date()), **resultado}


@app.get("/score/{customer_id}", response_model=ScoreResponse)
def score_get(customer_id: int, as_of_date: Optional[str] = None):
    """Variante GET de /score, para probar desde el navegador o un link directo."""
    return score(ScoreRequest(customer_id=customer_id, as_of_date=as_of_date))
