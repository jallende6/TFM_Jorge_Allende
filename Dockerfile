# Imagen del prototipo de scoring (TFM Fase 8). Sirve el modelo, no lo reentrena:
# sólo instala las dependencias de requirements-api.txt (FastAPI + sklearn/lightgbm
# para inferencia), no las de notebooks/entrenamiento (jupyter, xgboost, shap...).
#
# El dataset limpio (online_retail_II_cleaned.csv) NO está en git — es el histórico
# de transacciones sobre el que el prototipo calcula features en tiempo real, no un backup del set de entrenamiento. Antes de construir la
# imagen, generarlo localmente con notebooks/01_EDA_Online_Retail_II.ipynb (ver README.md).
#
# Build:  docker build -t tfm-scoring-api .
# Run:    docker run -p 8000:8000 tfm-scoring-api
# Docs:   http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# libgomp1: dependencia nativa de LightGBM (OpenMP) — sin ella el import falla
# con "libgomp.so.1: cannot open shared object file" al cargar models_fase3.pkl.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Módulo de features compartido con los notebooks: es el mismo archivo, no una
# copia — evita que la imagen sirva una lógica de features distinta de la que
# entrenó los modelos (training-serving skew, ver src/tfm_pipeline.py).
COPY src/ src/

# Prototipo visual (HTML/CSS/JS estático) servido por el propio api.py en "/".
COPY static/ static/

# Artefactos entrenados
COPY models/models_fase3.pkl models/
COPY models/models_clv.pkl models/

# Histórico necesario para calcular features de scoring en tiempo real
COPY data/processed/online_retail_II_cleaned.csv data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
