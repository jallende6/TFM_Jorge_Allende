# TFM · Analítica de lealtad del cliente — Online Retail II

github: https://github.com/jallende6/TFM_Jorge_Allende
video link: https://drive.google.com/file/d/1mqsP7-JW4TOvt-cCNfZh8k8xkZgbU9Ue/view?usp=sharing
Máster en Big Data, Data Science e IA (UCM) · Jorge Allende

Predicción de churn a 90 días, estimación de Customer Lifetime Value y segmentación
de clientes sobre el dataset [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
(UCI, CC BY 4.0), con un prototipo de scoring en tiempo real como cierre del trabajo.

## Estructura

```
notebooks/
  01_EDA_Online_Retail_II.ipynb        Fase 1 — limpieza y calidad de datos
  02_Feature_Engineering.ipynb         Fase 2 — RFM, target de churn, ventanas temporales
  03_Modeling.ipynb                    Fase 3 — churn: comparación de modelos
  03b_CLV.ipynb                        Fase 3b — CLV: comparación de modelos
  04_Interpretability.ipynb            Fase 4 — SHAP, perfiles de segmento
  05_Conclusions_Productization.ipynb  Fase 5 — prototipo de scoring + conclusiones

src/
  tfm_pipeline.py    Módulo compartido: features, preprocesado y regla de scoring.
                     Notebooks, src/api.py y .scripts/verificar_coherencia.py
                     importan de aquí — ninguno reimplementa esta lógica.
  api.py             API HTTP del prototipo (FastAPI). Envuelve tfm_pipeline. Se
                     importa como paquete: src.api:app (ver más abajo). Sirve
                     además el prototipo visual en "/" (static/index.html).

static/
  index.html         Prototipo visual: formulario Customer ID -> score, llama
                     al propio API por fetch() (mismo origen, sin servicio aparte).

data/
  raw/               Dataset original descargado de UCI (no versionado).
  processed/         Salidas de la Fase 1: dataset limpio, fechas de
                     observación, base RFM (no versionado el limpio, 96MB).
outputs/
  fase1_eda/         Salidas de la Fase 1 (calidad de datos, análisis temporal).
  fase2_features/    Salidas de la Fase 2 (features de cliente, documentación).
  fase3_modeling/    Salidas de la Fase 3 — churn (clusters, predicciones, ranking).
  fase3b_clv/        Salidas de la Fase 3b — CLV (matriz de prioridad, cuadrantes).
  fase4_interpretability/  Salidas de la Fase 4 (SHAP, coeficientes, resumen).
models/              Artefactos entrenados (.pkl). models_fase3.pkl y models_clv.pkl
                     consumen src/api.py y el notebook 05; interpretability_fase4.pkl
                     lo genera y consume el notebook 04.
```

Los notebooks viven en `notebooks/` y referencian `data/`, `outputs/` y `models/`
con rutas relativas `../` (Jupyter fija el directorio de trabajo del kernel en la
carpeta del notebook).

## Entrega

```
Jorge_Allende_Medrano_Analitica_Lealtad_Cliente/
  Jorge_Allende_TFM.pdf              Informe (10 secciones, límite 20 páginas)
  Jorge_Allende_TFM_Video.mp4        Vídeo de presentación
  Jorge_Allende_TFM_Anexos/
    Anexo_1_Prototipo/               API, interfaz y tfm_pipeline exportados a HTML
    Anexo_2_Codigo/                  Notebooks 01-05 exportados a HTML (código completo)
```

Los anexos no cuentan en el límite de 20 páginas del informe.

## Reproducir el análisis completo

**Requiere Python ≥ 3.11** y el dataset crudo, que no está en este repo (86MB, excluido
por tamaño). La cadena completa tarda unos 3 minutos en un portátil reciente.

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```


El paso 3 usa el intérprete del entorno activo y fuerza que **los seis notebooks se
ejecuten con el mismo**. Es importante: ejecutar unas fases en un entorno y otras en otro
desplaza el umbral de clasificación y desincroniza las fases posteriores sin producir
ningún error. 

Si prefieres ejecutarlos a mano en Jupyter, ábrelos desde `notebooks/` —usan rutas
relativas `../data`, `../outputs`, `../models`— en orden 01 → 05 y **con el mismo kernel
en todos**.

> **Sobre las versiones.** `requirements.txt` fija las del entorno en que se produjeron
> las cifras del informe. Con otras versiones el análisis sigue siendo válido y las
> cifras se desplazan ligeramente; el verificador lo avisa en lugar de fallar, siempre
> que toda la cadena se haya ejecutado en el mismo entorno.

El notebook 01 genera `data/processed/online_retail_II_cleaned.csv` (96MB, tampoco
versionado). Los notebooks 03/03b generan `models/models_fase3.pkl` y
`models/models_clv.pkl`, que sí están en el repo — no hace falta reentrenar para
levantar el prototipo o el API.

## Levantar el prototipo de scoring

### Opción A — directo con Python

Ejecutar desde la **raíz del repo** (api.py resuelve `data/` y `models/` como
rutas relativas a la raíz, y se importa como paquete `src.api`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn src.api:app --reload --port 8000
```

### Opción B — Docker (recomendado para probar sin tocar el entorno local)

```bash
docker build -t tfm-scoring-api .
docker run -p 8000:8000 tfm-scoring-api
```

Requiere `data/processed/online_retail_II_cleaned.csv` presente localmente antes
del build (ver "Reproducir el análisis completo", paso 3) — no está en git y el
Dockerfile lo copia a la imagen.

### Probar

```bash
curl -X POST http://localhost:8000/score \
     -H "Content-Type: application/json" \
     -d '{"customer_id": 16062}'
```

Prototipo visual: http://localhost:8000/

Documentación interactiva (Swagger UI): http://localhost:8000/docs
