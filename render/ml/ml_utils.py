# render/ml_utils.py
import os
import joblib
import numpy as np
import pandas as pd

from django.conf import settings

# Carga perezosa (se inicializan en el primer uso)
_MDL = None
_SX = None
_SY = None

def _model_paths():
    base = os.path.join(settings.BASE_DIR, "render", "ml")
    return (
        os.path.join(base, "consumo_intervalo.h5"),
        os.path.join(base, "scaler_X.pkl"),
        os.path.join(base, "scaler_y.pkl"),
    )

def _ensure_loaded():
    global _MDL, _SX, _SY
    if _MDL is None:
        from tensorflow.keras.models import load_model
        mdl_path, sx_path, sy_path = _model_paths()
        _MDL = load_model(mdl_path)
        _SX  = joblib.load(sx_path)
        _SY  = joblib.load(sy_path)

def predict_next_consumption_from_df(df: pd.DataFrame, lookback: int = 3) -> float:
    """
    df: DataFrame con columnas al menos: ts, flow_lpm, water_temp_c, humidity_pct.
         Debe venir ordenado por tiempo ascendente.
    Retorna: litros predichos para el siguiente intervalo (float).
    """
    _ensure_loaded()

    # Asegurar tipos y columnas
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"]).sort_values("ts")

    for c in ["flow_lpm", "water_temp_c", "humidity_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Aproximar consumo del último intervalo si no tienes “litros” ya calculado:
    # dt en minutos y litros = flow_lpm * dt_min
    df["dt_min"] = df["ts"].diff().dt.total_seconds().div(60).fillna(0).clip(lower=0, upper=60)
    df["litros"] = df["flow_lpm"] * df["dt_min"]

    if len(df) < lookback:
        raise ValueError(f"No hay suficientes muestras ({len(df)}) para lookback={lookback}")

    win = df.tail(lookback)
    x = np.array([[
        win["flow_lpm"].mean(),
        win["water_temp_c"].mean(),
        win["humidity_pct"].mean(),
        win["litros"].iloc[-1]           # último consumo estimado
    ]], dtype=np.float32)

    x_s  = _SX.transform(x)
    yhat = _MDL.predict(x_s, verbose=0)
    pred = _SY.inverse_transform(yhat)[0, 0]
    return float(max(pred, 0.0))
