# render/ml/ml_utils.py
import os
import numpy as np
import joblib
from django.conf import settings
import tflite_runtime.interpreter as tflite  # <-- SIN TensorFlow

# Carga perezosa (solo una vez)
_INTERPRETER = None
_INPUT_IDX = None
_OUTPUT_IDX = None
_SX = None
_SY = None


def _paths():
    base = os.path.join(settings.BASE_DIR, "render", "ml")
    return (
        os.path.join(base, "consumo_intervalo.tflite"),  # modelo TFLite
        os.path.join(base, "scaler_X.pkl"),              # MinMax/Standard para X
        os.path.join(base, "scaler_y.pkl"),              # MinMax/Standard para y
    )


def _ensure_loaded():
    """Inicializa intérprete TFLite e hidrata los escaladores .pkl (una vez)."""
    global _INTERPRETER, _INPUT_IDX, _OUTPUT_IDX, _SX, _SY
    if _INTERPRETER is not None:
        return

    tflite_path, sx_path, sy_path = _paths()

    # Intérprete TFLite (NO TensorFlow completo)
    _INTERPRETER = tflite.Interpreter(model_path=tflite_path)
    _INTERPRETER.allocate_tensors()

    _INPUT_IDX = _INTERPRETER.get_input_details()[0]["index"]
    _OUTPUT_IDX = _INTERPRETER.get_output_details()[0]["index"]

    # Escaladores
    _SX = joblib.load(sx_path)
    _SY = joblib.load(sy_path)


def predict_next_consumption(flow_lpm: float,
                             water_temp_c: float,
                             humidity_pct: float,
                             last_liters: float) -> float:
    """
    Devuelve la predicción (en litros) para el próximo intervalo usando TFLite.
    """
    _ensure_loaded()

    # Vector de entrada (4 features)
    x = np.array([[float(flow_lpm),
                   float(water_temp_c),
                   float(humidity_pct),
                   float(last_liters)]], dtype=np.float32)

    # Escalar igual que en el entrenamiento
    x_s = _SX.transform(x).astype(np.float32)

    # Ejecutar TFLite
    _INTERPRETER.set_tensor(_INPUT_IDX, x_s)
    _INTERPRETER.invoke()
    y_s = _INTERPRETER.get_tensor(_OUTPUT_IDX)

    # Volver a escala original
    y = _SY.inverse_transform(y_s)[0, 0]
    return float(max(y, 0.0))
