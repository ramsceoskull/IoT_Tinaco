# render/views.py
import json
from datetime import datetime, timezone
import requests
from django.core.paginator import Paginator
from django.shortcuts import render
import pandas as pd
from .ml.ml_utils import predict_next_consumption_from_df  

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
    prediction = None
    try:
        url = "https://iottinaco.onrender.com/readings/all"
        data = requests.get(url).json()

        df = pd.DataFrame(data)

        cols = ["ts", "flow_lpm", "water_temp_c", "humidity_pct"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0

        df = df.tail(3)

        prediction = predict_next_consumption_from_df(df)

    except Exception as e:
        prediction = f"Error: {e}"

    return render(request, "render/predict.html", {"prediction": prediction})
