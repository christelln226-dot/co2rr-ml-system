import joblib
import pandas as pd

deposition_model = joblib.load("models/deposition_model.pkl")
sputter_model = joblib.load("models/sputter_model.pkl")
electrochem_model = joblib.load("models/electrochem_model.pkl")

def predict_deposition(params: dict):
    X = pd.DataFrame([params])
    pred = deposition_model.predict(X)[0]
    return {
        "particle_size": float(pred[0]),
        "uniformity": float(pred[1]),
        "dispersion": float(pred[2]),
        "overgrowth": float(pred[3]),
    }

def predict_sputter(params: dict):
    X = pd.DataFrame([params])
    pred = sputter_model.predict(X)[0]
    return {
        "ptfe_thickness": float(pred[0]),
        "ptfe_uniformity": float(pred[1]),
        "film_quality": float(pred[2]),
    }

def predict_electrochem(params: dict):
    X = pd.DataFrame([params])
    pred = electrochem_model.predict(X)[0]
    return {
        "stability": float(pred[0]),
        "fe_co": float(pred[1]),
    }
