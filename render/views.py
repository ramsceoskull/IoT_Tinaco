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
    prediction = None

    # Obtener últimos datos desde tu API FastAPI
    api_url = "https://iottinaco.onrender.com/readings/latest"
    try:
        data = requests.get(api_url).json()
        flow = data.get("flow_lpm", 0)
        temp = data.get("waterTempC", 0)
        humidity = data.get("humidity_pct", 0)
        last_liters = data.get("litros_intervalo", 0)   # si lo agregan en API
        prediction = predict_consumption(flow, temp, humidity, last_liters)
    except:
        prediction = "Error obteniendo datos."

    return render(request, "predict.html", {"prediction": prediction})