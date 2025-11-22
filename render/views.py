import json

import requests
from django.core.paginator import Paginator
from django.shortcuts import render

import requests
from render.ml.ml_utils import predict_next_consumption

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
        # Obtener todas las lecturas
        url = "https://iottinaco.onrender.com/readings/all"
        data = requests.get(url).json()

        if not data or len(data) < 3:
            prediction = "No hay suficientes datos para predecir"
        else:
            # Tomar las últimas 3
            df = pd.DataFrame(data)
            df = df.tail(3)

            prediction = predict_next_consumption_from_df(df)

    except Exception as e:
        prediction = f"Error: {e}"

    return render(request, "render/predict.html", {
        "prediction": prediction
    })