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
    try:
        # Obtener los últimos datos reales
        data = requests.get("https://iottinaco.onrender.com/readings?limit=1").json()

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("La API no devolvió datos válidos")

        last = data[0]

        flow = float(last.get("flow_lpm", 0))
        temp = float(last.get("water_temp_c", 0))
        humidity = float(last.get("humidity_pct", 0))

        # calcular litros del último intervalo (si no existe)
        litros = float(last.get("litros_intervalo", 0))

        prediction = predict_next_consumption(flow, temp, humidity, litros)

    except Exception as e:
        prediction = f"Error: {e}"

    return render(request, "render/predict.html", {"prediction": prediction})
