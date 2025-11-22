import json

import requests
from django.core.paginator import Paginator
from django.shortcuts import render

import pandas as pd
import requests
from .ml_utils import predict_next_consumption_from_df

API_BASE = "https://iottinaco.onrender.com"  
DEVICE_ID = "TNR-01"

# Create your views here.
def index(request):
	url = 'https://iottinaco.onrender.com/readings/all'
	try:
		response = requests.get(url)
		response.raise_for_status()
		data = response.json()
	except requests.RequestException as e:
		data = []
		print(f"Error fetching data: {e}")

	# Paginación: 15 registros por página
	paginator = Paginator(data, 15)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	return render(request, "render/index.html", {"page_obj": page_obj})

def readings_chart(request):
	url = "https://iottinaco.onrender.com/readings/all"
	try:
			response = requests.get(url)
			response.raise_for_status()
			data = response.json()
	except requests.exceptions.RequestException as e:
			data = []
			print(f"Error al obtener datos: {e}")

	# Serializar correctamente para JavaScript
	data_json = json.dumps(data)

	return render(request, "render/readings_chart.html", {"readings_json": data_json})

def predict_view(request):
    """
    Construye un DataFrame con las últimas lecturas y llama al modelo TFLite
    vía predict_next_consumption_from_df(df). Muestra el próximo consumo estimado.
    """
    url = f"{API_BASE}/readings/all?device_id={DEVICE_ID}&limit=60&sort=asc"
    pred_liters = None
    msg = None

    try:
        rows = requests.get(url, timeout=10).json()
        if not rows:
            msg = "No hay datos suficientes."
        else:
            df = pd.DataFrame(rows)

            # Normaliza tipos y calcula litros del intervalo (si no tienes flow_lpm, puedes
            # comentar este bloque y usar la alternativa con delta de nivel)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            for c in ["flow_lpm", "water_temp_c", "humidity_pct", "level_pct"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # A) consumo desde flujo (preferido)
            if "flow_lpm" in df.columns:
                df["dt_min"] = df["ts"].diff().dt.total_seconds().div(60).fillna(0)
                df["dt_min"] = df["dt_min"].clip(lower=0, upper=60)
                df["litros"] = df["flow_lpm"].fillna(0) * df["dt_min"]
            else:
                # B) alternativa por cambio de nivel (ajusta V_TINACO)
                V_TINACO = 3000.0
                df["litros"] = (-(df["level_pct"].diff().fillna(0))/100.0) * V_TINACO
                df["litros"] = df["litros"].clip(lower=0)

            pred_liters = predict_next_consumption_from_df(df)  # <- usa tu util TFLite

    except Exception as e:
        msg = f"Error obteniendo o procesando datos: {e}"

    context = {"prediction": pred_liters, "message": msg}
    return render(request, "render/predict.html", context)