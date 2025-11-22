import json

import requests
from django.core.paginator import Paginator
from django.shortcuts import render

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

    try:
        # Obtener TODAS las lecturas
        data = requests.get("https://iottinaco.onrender.com/readings/all").json()

        if not data:
            raise ValueError("No hay datos disponibles")

        # Tomar el último registro
        latest = data[-1]

        flow = latest.get("flow_lpm", 0)
        temp = latest.get("water_temp_c", 0)
        humidity = latest.get("humidity_pct", 0)

        # Si no existe litros, lo calculamos rápido
        last_liters = flow * 1  # 1 minuto aprox (placeholder)

        prediction = predict_next_consumption_from_df(flow, temp, humidity, last_liters)

    except Exception as e:
        prediction = f"Error durante predicción: {e}"

    return render(request, "render/predict.html", {"prediction": prediction})