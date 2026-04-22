import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/simulated/full_simulated_data.csv")

X = df[[
    "particle_size", "uniformity", "dispersion", "overgrowth",
    "ptfe_thickness", "ptfe_uniformity",
    "current_density", "electrolyte_type", "electrolyte_conc"
]]
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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(
        RandomForestRegressor(n_estimators=250, random_state=42)
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=== 电化学性能模块评估 ===")
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f"{col}: MAE={mae:.3f}, R2={r2:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/electrochem_model.pkl")
print("电化学模型已保存：models/electrochem_model.pkl")
