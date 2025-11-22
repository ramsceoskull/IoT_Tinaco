# render/views.py
import json
from datetime import datetime, timezone
import requests
from django.core.paginator import Paginator
from django.shortcuts import render
import pandas as pd
from render.ml.ml_utils import predict_next_consumption  

API_BASE = "https://iottinaco.onrender.com"

def index(request):
    url = f"{API_BASE}/readings/all"
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException as e:
        data = []
        print(f"Error fetching data: {e}")

    paginator = Paginator(data, 15)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "render/index.html", {"page_obj": page_obj})

def readings_chart(request):
    url = f"{API_BASE}/readings/all"
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException as e:
        data = []
        print(f"Error al obtener datos: {e}")
    return render(request, "render/readings_chart.html", {"readings_json": json.dumps(data)})

def _iso_to_dt(s: str):
    
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def predict_view(request):
    """
    Obtiene lecturas de /readings/all, toma las últimas 3 y calcula
    el consumo del próximo intervalo con el modelo TFLite.
    """
    try:
        # 1) Traer TODO y filtrar aquí (evita el 422 del ?limit=2)
        resp = requests.get(f"{API_BASE}/readings/all", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 3:
            raise ValueError("La API no devolvió datos suficientes (>=3).")

        # 2) DataFrame y limpieza mínima
        df = pd.DataFrame(data)
        if "ts" not in df.columns:
            raise ValueError("La API no devolvió la columna 'ts'.")

        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"]).sort_values("ts")

        # asegurar columnas numéricas
        for c in ["flow_lpm", "water_temp_c", "humidity_pct", "level_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 3) Aproximar litros del último intervalo si no viene pre-calculado
        #    Usa flujo si existe, si no, deriva de level_pct con un volumen estimado
        df["dt_min"] = df["ts"].diff().dt.total_seconds().div(60).fillna(0)
        df["dt_min"] = df["dt_min"].clip(lower=0, upper=60)

        if "flow_lpm" in df.columns and df["flow_lpm"].notna().any():
            df["litros_intervalo"] = df["flow_lpm"].fillna(0) * df["dt_min"]
        else:
            V_TINACO = 3000.0  # litros estimados del tinaco
            df["litros_intervalo"] = (-(df["level_pct"].diff().fillna(0))/100.0) * V_TINACO
            df["litros_intervalo"] = df["litros_intervalo"].clip(lower=0)

        # 4) Tomar la ventana reciente (3 muestras)
        last = df.tail(3)
        flow     = float(last["flow_lpm"].fillna(0).mean()) if "flow_lpm" in df.columns else 0.0
        temp     = float(last["water_temp_c"].fillna(0).mean()) if "water_temp_c" in df.columns else 0.0
        humidity = float(last["humidity_pct"].fillna(0).mean()) if "humidity_pct" in df.columns else 0.0
        last_liters = float(last["litros_intervalo"].iloc[-1])

        # 5) Predicción con TFLite
        pred_l = predict_next_consumption(flow, temp, humidity, last_liters)

        ctx = {
            "prediction": f"{pred_l:.2f}",
            "units": "L"
        }
        return render(request, "render/predict.html", ctx)

    except Exception as e:
        # Mensaje claro en la vista
        return render(request, "render/predict.html", {
            "prediction": f"Error: {e}",
            "units": "L"
        })
