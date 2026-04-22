import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/simulated/full_simulated_data.csv"
DEPOSITION_MODEL_PATH = "models/deposition_model.pkl"
SPUTTER_MODEL_PATH = "models/sputter_model.pkl"
ELECTROCHEM_MODEL_PATH = "models/electrochem_model.pkl"


def train_deposition_model():
    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "current", "ton", "toff", "cycles",
            "agno3", "kno3", "pvp", "citrate"
        ]
    ]
    y = df[
        [
            "particle_size", "uniformity", "dispersion", "overgrowth"
        ]
    ]

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=250, random_state=42)
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, DEPOSITION_MODEL_PATH)
    return model


def train_sputter_model():
    df = pd.read_csv(DATA_PATH)

    X = df[["power", "pressure", "sputter_time", "distance"]]
    y = df[["ptfe_thickness", "ptfe_uniformity", "film_quality"]]

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=250, random_state=42)
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, SPUTTER_MODEL_PATH)
    return model


def train_electrochem_model():
    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "particle_size", "uniformity", "dispersion", "overgrowth",
            "ptfe_thickness", "ptfe_uniformity",
            "current_density", "electrolyte_type", "electrolyte_conc"
        ]
    ]
    y = df[["stability", "fe_co"]]

    numeric_features = [
        "particle_size", "uniformity", "dispersion", "overgrowth",
        "ptfe_thickness", "ptfe_uniformity",
        "current_density", "electrolyte_conc"
    ]
    categorical_features = ["electrolyte_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                MultiOutputRegressor(
                    RandomForestRegressor(n_estimators=250, random_state=42)
                ),
            ),
        ]
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, ELECTROCHEM_MODEL_PATH)
    return model


def safe_load_or_train(model_path, trainer_func):
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return trainer_func()
    except Exception as e:
        print(f"加载模型失败，改为重新训练: {model_path}")
        print(f"具体原因: {e}")
        return trainer_func()


deposition_model = safe_load_or_train(DEPOSITION_MODEL_PATH, train_deposition_model)
sputter_model = safe_load_or_train(SPUTTER_MODEL_PATH, train_sputter_model)
electrochem_model = safe_load_or_train(ELECTROCHEM_MODEL_PATH, train_electrochem_model)


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