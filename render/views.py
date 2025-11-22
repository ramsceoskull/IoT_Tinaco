# render/views.py
import json
from datetime import datetime, timezone
import requests
from django.core.paginator import Paginator
from django.shortcuts import render

from .ml.ml_utils import predict_next_consumption  # <- usa el TFLite

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
    # convierte "2025-11-12T03:03:53.683329Z" a datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def predict_view(request):
    """
    Toma las últimas 2 lecturas para estimar 'last_liters' con flow_lpm*Δt
    y predice el consumo del siguiente intervalo con el modelo TFLite.
    """
    try:
        # Trae 2 últimas lecturas (ajusta si tu API usa otro parámetro)
        resp = requests.get(f"{API_BASE}/readings?limit=2", timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            raise RuntimeError("No hay lecturas disponibles.")

        # Lectura más reciente
        r0 = rows[0]
        flow = float(r0.get("flow_lpm", 0) or 0)
        temp = float(r0.get("water_temp_c", 0) or 0)
        hum  = float(r0.get("humidity_pct", 0) or 0)

        # Estima litros del último intervalo (si hay dos timestamps)
        last_liters = 0.0
        if len(rows) >= 2 and r0.get("ts") and rows[1].get("ts"):
            t0 = _iso_to_dt(r0["ts"])
            t1 = _iso_to_dt(rows[1]["ts"])
            dt_min = max(0.0, min(60.0, (t0 - t1).total_seconds() / 60.0))
            last_liters = flow * dt_min

        pred_liters = predict_next_consumption(flow, temp, hum, last_liters)
        return render(request, "render/predict.html", {"prediction": f"{pred_liters:.2f} L"})

    except Exception as e:
        return render(request, "render/predict.html", {"prediction": f"Error: {e} L"})
