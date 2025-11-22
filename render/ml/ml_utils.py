# render/ml/ml_utils.py
import os
import numpy as np
import joblib
from django.conf import settings

# Carga perezosa (solo una vez)
_INTERPRETER = None
_INPUT_IDX   = 0
_OUTPUT_IDX  = 0
_SX = None
_SY = None

def _paths():
    base = os.path.join(settings.BASE_DIR, "render", "ml")
    return (
        os.path.join(base, "consumo_intervalo.tflite"),
        os.path.join(base, "scaler_X.pkl"),
        os.path.join(base, "scaler_y.pkl"),
    )

def _ensure_loaded():
    global _INTERPRETER, _INPUT_IDX, _OUTPUT_IDX, _SX, _SY
    if _INTERPRETER is None:
        import tensorflow as tf  # usa TF completo si existe
        tflite_path, sx_path, sy_path = _paths()
        _INTERPRETER = tf.lite.Interpreter(model_path=tflite_path)
        _INTERPRETER.allocate_tensors()
        _INPUT_IDX  = _INTERPRETER.get_input_details()[0]["index"]
        _OUTPUT_IDX = _INTERPRETER.get_output_details()[0]["index"]
        _SX = joblib.load(sx_path)
        _SY = joblib.load(sy_path)

def predict_next_consumption(flow_lpm: float, water_temp_c: float,
                             humidity_pct: float, last_liters: float) -> float:
    """
    Devuelve la predicción de litros para el próximo intervalo.
    """
    _ensure_loaded()
    x = np.array([[float(flow_lpm), float(water_temp_c),
                   float(humidity_pct), float(last_liters)]], dtype=np.float32)
    x_s = _SX.transform(x).astype(np.float32)

    _INTERPRETER.set_tensor(_INPUT_IDX, x_s)
    _INTERPRETER.invoke()
    yhat_s = _INTERPRETER.get_tensor(_OUTPUT_IDX)
    yhat   = _SY.inverse_transform(yhat_s)[0, 0]
    return float(max(yhat, 0.0))
