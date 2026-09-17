"""Lógica compartida del pipeline de lealtad del cliente (Online Retail II).

Este módulo existe para que la construcción de features no pueda divergir entre
notebooks. La Fase 2 la usa para generar el dataset de modelización y la Fase 5
la usa para scorear un cliente en producción: ambas llaman a la MISMA función.

Cualquier cambio en la definición de una feature se propaga automáticamente a
todo el pipeline, incluido el prototipo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Parámetros del esquema temporal -----------------------------------------
# Definidos aquí y leídos por todos los notebooks: no se redeclaran a mano.
LOOKBACK_MONTHS = 6    # profundidad del histórico usado para features
HORIZON_DAYS = 90      # horizonte de predicción (churn y CLV)

# --- Paleta de los cuadrantes de prioridad ------------------------------------
# Indexada por el NÚMERO de cuadrante, no por posición en la tabla: el color de
# «RETENER» no puede depender de cómo venga ordenado el DataFrame. Las Fases 3b y 5
# dibujan el mismo gráfico y deben pintarlo igual.
COLORES_CUADRANTE = {
    '1': '#d62728',   # RETENER    — alto riesgo, alto valor
    '2': '#2ca02c',   # FIDELIZAR  — bajo riesgo, alto valor
    '3': '#ff7f0e',   # DEJAR IR   — alto riesgo, bajo valor
    '4': '#7f7f7f',   # MONITOREAR — bajo riesgo, bajo valor
}


def setup_notebook(seaborn_style: bool = True) -> None:
    """Configuración común de los notebooks: filtros de avisos y estilo de gráficos.

    Vivía duplicada en la primera celda de los seis notebooks (~90 líneas repetidas),
    y ya había empezado a divergir entre ellos. Al estar aquí, cambiar un filtro es
    un solo cambio y no puede desincronizarse entre fases.

    Filtros acotados y justificados. NO se usa un 'ignore' global: en pandas 2.x
    varios FutureWarning anuncian cambios que alteran el resultado, no solo la salida.
      1) numpy 2.x compilado contra Accelerate (macOS) emite RuntimeWarning espurios
         en `matmul`. Se reproducen con matrices aleatorias limpias, sin inf ni NaN,
         y no afectan al resultado: es un artefacto del backend BLAS, no del modelo.
      2) Glifos ausentes en la fuente al renderizar el símbolo £ en algunos gráficos.

    Args:
        seaborn_style: aplica el tema `whitegrid`. La Fase 5 no usa seaborn y pasa
            False para conservar el estilo con el que se generaron sus gráficos.
    """
    import os
    import warnings

    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message='.*encountered in matmul')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

    # Los workers de joblib (n_jobs=-1) son procesos nuevos y no heredan los filtros
    # del proceso padre: el mismo aviso hay que silenciarlo por entorno, y sólo para
    # RuntimeWarning. FutureWarning y DeprecationWarning siguen siendo visibles.
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::RuntimeWarning')

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['font.size'] = 10

    if seaborn_style:
        import seaborn as sns
        sns.set_style('whitegrid')

# --- Contrato de features -----------------------------------------------------
# FEATURES entra en los modelos. DESCRIPTIVAS se calculan y se guardan por su
# valor de perfilado, pero se excluyen por redundancia algebraica.
FEATURES = [
    'recency_days',
    'frequency',
    'monetary',
    'return_rate',
    'customer_age_days',
    'avg_items_per_transaction',
    'unique_products',
]
DESCRIPTIVAS = ['avg_order_value']
LOG_FEATURES = ['frequency', 'monetary', 'avg_items_per_transaction', 'unique_products']
TARGETS = ['churn', 'future_monetary_90d']
KEYS = ['CustomerID', 'observation_date', 'window_month_year']


def preprocess(X: pd.DataFrame, log_features=None) -> pd.DataFrame:
    """log1p sobre las features de cola pesada. Idéntico en train, test y producción.

    Definida una sola vez: las Fases 3, 3b, 4 y 5 la importan en lugar de
    redefinirla, de modo que no puede desincronizarse.
    """
    log_features = LOG_FEATURES if log_features is None else log_features
    X = X.copy()
    for col in log_features:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X


def _aggregate_features(past: pd.DataFrame, observation_date: pd.Timestamp) -> pd.DataFrame:
    """Núcleo de cálculo de features. Recibe las transacciones de [t-lookback, t).

    Población: clientes con al menos una LÍNEA DE COMPRA (Quantity > 0) en la
    ventana. Un cliente cuya única actividad fue devolver no ha comprado, y por
    tanto no forma parte de la base sobre la que se predice retención.

    Separación de señales:
      - `frequency`, `monetary`, `unique_products` y `total_quantity` se calculan
        SOLO sobre líneas de compra. Las facturas de cancelación (`Invoice`
        empieza por C) no cuentan como compras.
      - Las devoluciones se capturan aparte, en `return_rate`, sobre el total de
        facturas del cliente en la ventana.
    """
    compras = past[past['Quantity'] > 0]
    if compras.empty:
        return pd.DataFrame()

    g = compras.groupby('Customer ID')
    w = pd.DataFrame({
        'first_purchase': g['InvoiceDate'].min(),
        'last_purchase': g['InvoiceDate'].max(),
        'frequency': g['Invoice'].nunique(),
        # Los regalos/muestras (Price = 0) son líneas de compra y suman £0.
        'monetary': g['Amount'].sum(),
        'total_quantity': g['Quantity'].sum(),
        'unique_products': g['StockCode'].nunique(),
    })

    # Devoluciones: proporción de FACTURAS con al menos una línea devuelta,
    # sobre el total de facturas del cliente (compras + devoluciones). Acotada a [0, 1].
    facturas_totales = past.groupby('Customer ID')['Invoice'].nunique()
    facturas_devolucion = (past[past['is_return'] == 1]
                           .groupby('Customer ID')['Invoice'].nunique())

    w['total_invoices'] = facturas_totales.reindex(w.index).fillna(0)
    w['returning_invoices'] = facturas_devolucion.reindex(w.index).fillna(0)

    w['recency_days'] = (observation_date - w['last_purchase']).dt.days
    w['customer_age_days'] = (w['last_purchase'] - w['first_purchase']).dt.days
    w['return_rate'] = w['returning_invoices'] / w['total_invoices']
    w['avg_order_value'] = w['monetary'] / w['frequency']
    w['avg_items_per_transaction'] = w['total_quantity'] / w['frequency']

    return w


def build_window(df: pd.DataFrame,
                 observation_date: pd.Timestamp,
                 lookback_months: int = LOOKBACK_MONTHS,
                 horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Features del pasado + targets del futuro para una fecha de observación.

    Args:
        df: transacciones limpias (Fase 1).
        observation_date: fecha de corte t.
        lookback_months: profundidad del histórico para features.
        horizon_days: horizonte futuro para churn y future_monetary.

    Returns:
        DataFrame a nivel cliente. Población = clientes con al menos una compra
        en [t - lookback, t); un cliente inactivo no se puede scorear ni retener.

    Los dos targets son complementarios por construcción:
        churn == 1  <=>  future_monetary_90d == 0
    Ambos se definen sobre COMPRAS en [t, t+horizon): una devolución en la
    ventana futura no cuenta como que el cliente sigue activo.
    """
    hist_start = observation_date - pd.DateOffset(months=lookback_months)
    future_end = observation_date + pd.Timedelta(days=horizon_days)

    past = df[(df['InvoiceDate'] >= hist_start) & (df['InvoiceDate'] < observation_date)]
    future = df[(df['InvoiceDate'] >= observation_date) & (df['InvoiceDate'] < future_end)]

    w = _aggregate_features(past, observation_date)
    if w.empty:
        return pd.DataFrame()

    # --- Targets (solo información posterior a t) ---
    compras_futuras = future[future['Quantity'] > 0]
    gasto_futuro = (compras_futuras.groupby('Customer ID')['Amount'].sum()
                    .reindex(w.index).fillna(0))

    w['future_monetary_90d'] = gasto_futuro
    w['churn'] = (gasto_futuro <= 0).astype(int)

    w['observation_date'] = observation_date
    w['window_month_year'] = observation_date.strftime('%Y-%m')

    return w.reset_index().rename(columns={'Customer ID': 'CustomerID'})


def customer_features(transacciones: pd.DataFrame,
                      customer_id: int,
                      as_of_date,
                      lookback_months: int = LOOKBACK_MONTHS) -> dict | None:
    """Features de UN cliente a una fecha arbitraria, para scoring en producción.

    Reutiliza `_aggregate_features`, el mismo núcleo que `build_window`: el
    prototipo de la Fase 5 no puede calcular las features de otra forma que el
    entrenamiento. Devuelve None si el cliente no tiene compras en la ventana.
    """
    as_of_date = pd.to_datetime(as_of_date)
    inicio = as_of_date - pd.DateOffset(months=lookback_months)

    past = transacciones[(transacciones['Customer ID'] == customer_id) &
                         (transacciones['InvoiceDate'] >= inicio) &
                         (transacciones['InvoiceDate'] < as_of_date)]

    w = _aggregate_features(past, as_of_date)
    if w.empty:
        return None

    return {col: w.iloc[0][col] for col in FEATURES}


# --- Regla de decisión (scoring) -----------------------------------------------
# Vivía en el notebook 05 hasta que api.py necesitó la misma lógica: una API HTTP
# reimplementándola habría sido exactamente el training-serving skew que el
# módulo existe para evitar (ver docstring del archivo). Un solo lugar, tres
# consumidores: el notebook 05, api.py y verificar_coherencia.py.
ACCIONES = {
    'RETENER': 'Contacto personalizado con oferta. Es donde cada libra de presupuesto rinde más.',
    'FIDELIZAR': 'Programa de valor añadido: acceso anticipado, condiciones preferentes. No está '
                 'en riesgo, pero es demasiado valioso para desatender.',
    'DEJAR IR': 'Sólo canales automáticos. Una campaña personalizada costaría más de lo que aporta.',
    'MONITOREAR': 'Sin acción. Revisar en el siguiente ciclo.',
}


def score_customer(features: dict, fase3: dict, faseclv: dict) -> dict:
    """Features -> riesgo de churn, valor esperado y decisión de retención.

    Recibe los diccionarios completos deserializados de `models_fase3.pkl` y
    `models_clv.pkl` (el mismo contrato que ya consume la Fase 5) en lugar de
    variables globales, para que esta función se pueda llamar igual desde un
    notebook, desde `api.py` o desde un test, sin depender de qué haya en el
    entorno que la rodea.
    """
    modelo_churn = fase3[f"{fase3['best_model_key']}_best"]
    modelo_clv = faseclv[f"{faseclv['best_model_key']}_best"]

    X = preprocess(pd.DataFrame([features])[FEATURES])

    riesgo = float(modelo_churn.predict_proba(fase3['scaler'].transform(X))[0, 1])

    # Mismo destransformado que en la Fase 3b: expm1 -> smearing -> tope del máximo visto.
    clv_log = modelo_clv.predict(faseclv['scaler_clv'].transform(X))[0]
    clv = float(np.clip(np.expm1(clv_log) * faseclv['smearing'], 0, faseclv['gasto_max_train']))

    alto_riesgo = riesgo >= fase3['threshold']
    alto_valor = clv >= faseclv['clv_median']
    decision = ('RETENER' if alto_riesgo and alto_valor else
                'FIDELIZAR' if alto_valor else
                'DEJAR IR' if alto_riesgo else 'MONITOREAR')

    return {
        'riesgo_churn_90d': round(riesgo, 3),
        'clv_si_vuelve': round(clv, 2),
        'valor_esperado': round((1 - riesgo) * clv, 2),
        'valor_en_riesgo': round(riesgo * clv, 2),
        'decision': decision,
        'accion': ACCIONES[decision],
    }


def score_customer_from_transactions(transacciones: pd.DataFrame,
                                     customer_id: int,
                                     as_of_date,
                                     fase3: dict,
                                     faseclv: dict,
                                     lookback_months: int = LOOKBACK_MONTHS) -> dict | None:
    """`customer_features()` + `score_customer()` en un solo paso.

    None si el cliente no tiene compras en la ventana (no es scoreable) — mismo
    criterio de población que usó el entrenamiento.
    """
    features = customer_features(transacciones, customer_id, as_of_date, lookback_months)
    if features is None:
        return None
    return {**score_customer(features, fase3, faseclv), '_features': features}


def observation_calendar(df: pd.DataFrame,
                         lookback_months: int = LOOKBACK_MONTHS,
                         horizon_days: int = HORIZON_DAYS) -> pd.DatetimeIndex:
    """Fechas de observación mensuales compatibles con los datos disponibles.

    La primera ventana necesita `lookback_months` completos por detrás y la
    última `horizon_days` completos por delante. La primera fecha se redondea
    hacia ARRIBA al inicio de mes y la última hacia ABAJO, de modo que ninguna
    ventana quede incompleta aunque el dataset no empiece en día 1.

    La cobertura se evalúa a nivel de DÍA (`normalize`), no de instante: el primer
    día con datos cuenta como completo aunque su primera transacción sea a las
    07:45. Comparar contra la hora exacta descartaría una ventana entera por unas
    horas en las que, por definición, no hubo ninguna venta.
    """
    primero_valido = (df['InvoiceDate'].min().normalize()
                      + pd.DateOffset(months=lookback_months))
    ultimo_valido = (df['InvoiceDate'].max().normalize()
                     - pd.Timedelta(days=horizon_days))

    primera = primero_valido.normalize().replace(day=1)
    if primera < primero_valido:                      # redondeo hacia arriba
        primera = primera + pd.DateOffset(months=1)
    ultima = ultimo_valido.normalize().replace(day=1)  # redondeo hacia abajo

    return pd.date_range(primera, ultima, freq='MS')
