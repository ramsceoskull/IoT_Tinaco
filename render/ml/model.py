import numpy as np
import joblib
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo TFLite y scalers
TFLITE_PATH = os.path.join(BASE_DIR, "consumo_intervalo.tflite")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")

# Cargar TFLite
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar scalers
scaler_X = joblib.load(SCALER_X_PATH)
scaler_Y = joblib.load(SCALER_Y_PATH)

def predict_consumption(flow, temp, humidity, last_liters):
    """Recibe últimas variables y predice consumo."""
    x = np.array([[flow, temp, humidity, last_liters]], dtype=np.float32)

    x_scaled = scaler_X.transform(x)

    interpreter.set_tensor(input_details[0]['index'], x_scaled)
    interpreter.invoke()

    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    y = scaler_Y.inverse_transform(y_scaled)

    return float(y[0][0])
