import os
from pathlib import Path
import textwrap

BASE = Path.cwd()

folders = [
    "data/simulated",
    "data/raw",
    "data/raw/sem_images",
    "data/processed",
    "models",
    "src",
    "outputs",
]
for folder in folders:
    (BASE / folder).mkdir(parents=True, exist_ok=True)

files = {}

files["requirements.txt"] = """
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
opencv-python
scipy
joblib
pillow
xgboost
lightgbm
openpyxl
"""

files["src/generate_simulated_data.py"] = r'''
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 600

# -------------------------
# 模块二：电沉积参数
# -------------------------
current = np.random.uniform(1, 20, N)          # mA
ton = np.random.uniform(0.01, 1.0, N)          # s
toff = np.random.uniform(0.01, 2.0, N)         # s
cycles = np.random.randint(10, 500, N)

agno3 = np.random.uniform(0.001, 0.1, N)       # M
kno3 = np.random.uniform(0.01, 1.0, N)         # M
pvp = np.random.uniform(0.0, 5.0, N)           # g/L
citrate = np.random.uniform(0.0, 0.1, N)       # M

particle_size = (
    20
    + 2.3 * current
    + 9 * ton
    - 4 * toff
    + 0.03 * cycles
    + 45 * agno3
    - 1.8 * pvp
    - 25 * citrate
    + np.random.normal(0, 3, N)
)

uniformity = (
    82
    - 1.0 * current
    - 3.5 * ton
    + 2.5 * toff
    - 0.015 * cycles
    - 18 * agno3
    + 2.0 * pvp
    + 8 * citrate
    + np.random.normal(0, 2, N)
)

dispersion = (
    72
    - 1.4 * current
    + 2.3 * toff
    + 1.2 * pvp
    + np.random.normal(0, 2, N)
)

overgrowth = (
    0.4 * current
    + 7 * ton
    + 0.02 * cycles
    + 26 * agno3
    - 1.5 * pvp
    + np.random.normal(0, 2, N)
)

# -------------------------
# 模块三：PTFE 溅射
# -------------------------
power = np.random.uniform(20, 150, N)          # W
pressure = np.random.uniform(0.1, 2.0, N)      # Pa
sputter_time = np.random.uniform(30, 600, N)   # s
distance = np.random.uniform(3, 10, N)         # cm

ptfe_thickness = (
    5
    + 0.22 * power
    + 0.055 * sputter_time
    - 2.8 * pressure
    - 1.1 * distance
    + np.random.normal(0, 3, N)
)

ptfe_uniformity = (
    76
    - 0.07 * power
    - 0.012 * sputter_time
    - 2.5 * pressure
    - 1.3 * distance
    + np.random.normal(0, 2, N)
)

film_quality = (
    60
    + 0.11 * power
    + 0.03 * sputter_time
    - 2.1 * pressure
    - 1.0 * distance
    + np.random.normal(0, 2, N)
)

# -------------------------
# 模块四：电化学测试
# -------------------------
current_density = np.random.uniform(50, 500, N)
electrolyte_type = np.random.choice(["K2CO3", "KHCO3", "KOH", "H2O"], N)
electrolyte_conc = np.random.uniform(0.1, 3.0, N)

electrolyte_map = {
    "K2CO3": 1.0,
    "KHCO3": 0.85,
    "KOH": 1.15,
    "H2O": 0.55
}
electrolyte_factor = np.array([electrolyte_map[x] for x in electrolyte_type])

stability = (
    18
    + 0.55 * uniformity
    + 0.35 * dispersion
    - 0.28 * particle_size
    - 0.45 * overgrowth
    + 0.18 * ptfe_uniformity
    + 0.12 * ptfe_thickness
    - 0.028 * current_density
    + 5.5 * electrolyte_factor
    + 1.8 * electrolyte_conc
    + np.random.normal(0, 5, N)
)

fe_co = (
    28
    + 0.30 * uniformity
    + 0.22 * dispersion
    - 0.14 * particle_size
    - 0.20 * overgrowth
    + 0.24 * ptfe_uniformity
    + 0.09 * ptfe_thickness
    - 0.018 * current_density
    + 7.5 * electrolyte_factor
    + 1.5 * electrolyte_conc
    + np.random.normal(0, 4, N)
)

df = pd.DataFrame({
    "sample_id": [f"S{i+1:04d}" for i in range(N)],

    "current": current,
    "ton": ton,
    "toff": toff,
    "cycles": cycles,
    "agno3": agno3,
    "kno3": kno3,
    "pvp": pvp,
    "citrate": citrate,

    "particle_size": particle_size,
    "uniformity": uniformity,
    "dispersion": dispersion,
    "overgrowth": overgrowth,

    "power": power,
    "pressure": pressure,
    "sputter_time": sputter_time,
    "distance": distance,

    "ptfe_thickness": ptfe_thickness,
    "ptfe_uniformity": ptfe_uniformity,
    "film_quality": film_quality,

    "current_density": current_density,
    "electrolyte_type": electrolyte_type,
    "electrolyte_conc": electrolyte_conc,

    "stability": stability,
    "fe_co": fe_co
})

os.makedirs("data/simulated", exist_ok=True)
df.to_csv("data/simulated/full_simulated_data.csv", index=False)
print("模拟数据已生成：data/simulated/full_simulated_data.csv")
'''

files["src/train_deposition.py"] = r'''
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/simulated/full_simulated_data.csv")

X = df[[
    "current", "ton", "toff", "cycles",
    "agno3", "kno3", "pvp", "citrate"
]]
y = df[[
    "particle_size", "uniformity", "dispersion", "overgrowth"
]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=250, random_state=42)
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=== 电沉积模块评估 ===")
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f"{col}: MAE={mae:.3f}, R2={r2:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/deposition_model.pkl")
print("电沉积模型已保存：models/deposition_model.pkl")
'''

files["src/train_sputter.py"] = r'''
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/simulated/full_simulated_data.csv")

X = df[["power", "pressure", "sputter_time", "distance"]]
y = df[["ptfe_thickness", "ptfe_uniformity", "film_quality"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=250, random_state=42)
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=== PTFE 溅射模块评估 ===")
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f"{col}: MAE={mae:.3f}, R2={r2:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sputter_model.pkl")
print("溅射模型已保存：models/sputter_model.pkl")
'''

files["src/train_electrochem.py"] = r'''
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
'''

files["src/predict.py"] = r'''
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
'''

files["src/sem_analysis.py"] = r'''
import os
import cv2
import numpy as np
import pandas as pd

def analyze_sem_array(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    particle_areas = []
    centers = []
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue

        particle_areas.append(area)
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 1)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centers.append((cx, cy))
            cv2.circle(overlay, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    if len(particle_areas) == 0:
        result = {
            "coverage": 0.0,
            "particle_count": 0,
            "mean_particle_size": 0.0,
            "size_std": 0.0,
            "uniformity": 0.0,
            "dispersion": 0.0,
            "overgrowth_index": 0.0,
        }
        return result, binary, overlay, np.array([])

    particle_areas = np.array(particle_areas)
    eq_diameters = np.sqrt(4 * particle_areas / np.pi)

    coverage = np.sum(particle_areas) / (img_gray.shape[0] * img_gray.shape[1]) * 100
    particle_count = len(particle_areas)
    mean_particle_size = np.mean(eq_diameters)
    size_std = np.std(eq_diameters)

    uniformity = 100 / (1 + size_std)

    if len(centers) > 1:
        centers = np.array(centers)
        nearest = []
        for i in range(len(centers)):
            diff = centers - centers[i]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            dist = dist[dist > 0]
            if len(dist) > 0:
                nearest.append(np.min(dist))
        dispersion = float(np.mean(nearest)) if len(nearest) else 0.0
    else:
        dispersion = 0.0

    overgrowth_index = float(
        np.sum(eq_diameters > np.mean(eq_diameters) * 1.5) / len(eq_diameters)
    )

    result = {
        "coverage": float(coverage),
        "particle_count": int(particle_count),
        "mean_particle_size": float(mean_particle_size),
        "size_std": float(size_std),
        "uniformity": float(uniformity),
        "dispersion": float(dispersion),
        "overgrowth_index": float(overgrowth_index),
    }
    return result, binary, overlay, eq_diameters

def analyze_sem_image(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return analyze_sem_array(img_gray)

def batch_extract_sem_features(image_folder, output_csv="data/processed/sem_features.csv"):
    rows = []
    image_folder = os.path.abspath(image_folder)

    for name in os.listdir(image_folder):
        lower = name.lower()
        if not lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            continue

        image_path = os.path.join(image_folder, name)
        result, _, _, _ = analyze_sem_image(image_path)
        sample_id = os.path.splitext(name)[0]

        row = {"sample_id": sample_id, "image_path": image_path}
        row.update(result)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"SEM 特征已保存到: {output_csv}")
    return df
'''

files["src/optimize.py"] = r'''
import pandas as pd
import numpy as np
from itertools import product
from src.predict import predict_deposition, predict_sputter, predict_electrochem

def recommend_next_experiments(top_k=5):
    current_list = [5, 8, 10, 12, 15]
    ton_list = [0.05, 0.1, 0.2, 0.3]
    toff_list = [0.2, 0.5, 1.0]
    cycles_list = [50, 100, 200]
    agno3_list = [0.005, 0.01, 0.02]
    kno3_list = [0.05, 0.1, 0.2]
    pvp_list = [0.5, 1.0, 2.0]
    citrate_list = [0.005, 0.01, 0.02]

    power_list = [40, 60, 80]
    pressure_list = [0.3, 0.5, 1.0]
    sputter_time_list = [60, 120, 180]
    distance_list = [4, 5, 6]

    current_density_list = [100, 200, 300]
    electrolyte_type_list = ["K2CO3", "KHCO3", "KOH"]
    electrolyte_conc_list = [0.5, 1.0, 2.0]

    all_rows = []

    dep_space = product(
        current_list, ton_list, toff_list, cycles_list,
        agno3_list, kno3_list, pvp_list, citrate_list
    )
    sputter_space = product(
        power_list, pressure_list, sputter_time_list, distance_list
    )
    elec_space = product(
        current_density_list, electrolyte_type_list, electrolyte_conc_list
    )

    dep_candidates = list(dep_space)[:200]
    sputter_candidates = list(sputter_space)[:40]
    elec_candidates = list(elec_space)

    for dep in dep_candidates:
        dep_params = {
            "current": dep[0], "ton": dep[1], "toff": dep[2], "cycles": dep[3],
            "agno3": dep[4], "kno3": dep[5], "pvp": dep[6], "citrate": dep[7]
        }
        dep_result = predict_deposition(dep_params)

        for sp in sputter_candidates:
            sputter_params = {
                "power": sp[0], "pressure": sp[1], "sputter_time": sp[2], "distance": sp[3]
            }
            sputter_result = predict_sputter(sputter_params)

            for el in elec_space:
                electrochem_params = {
                    "particle_size": dep_result["particle_size"],
                    "uniformity": dep_result["uniformity"],
                    "dispersion": dep_result["dispersion"],
                    "overgrowth": dep_result["overgrowth"],
                    "ptfe_thickness": sputter_result["ptfe_thickness"],
                    "ptfe_uniformity": sputter_result["ptfe_uniformity"],
                    "current_density": el[0],
                    "electrolyte_type": el[1],
                    "electrolyte_conc": el[2]
                }
                elec_result = predict_electrochem(electrochem_params)

                score = 0.55 * elec_result["fe_co"] + 0.45 * elec_result["stability"]

                row = {}
                row.update(dep_params)
                row.update(sputter_params)
                row.update({
                    "current_density": el[0],
                    "electrolyte_type": el[1],
                    "electrolyte_conc": el[2],
                    "pred_stability": elec_result["stability"],
                    "pred_fe_co": elec_result["fe_co"],
                    "score": score
                })
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return df
'''

files["app.py"] = r'''
import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.predict import predict_deposition, predict_sputter, predict_electrochem
from src.sem_analysis import analyze_sem_array
from src.optimize import recommend_next_experiments

st.set_page_config(page_title="CO2RR 智能预测平台", layout="wide")
st.title("CO2RR 气体扩散电极智能预测平台")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "模块1 SEM图像分析",
    "模块2 电沉积预测",
    "模块3 PTFE溅射预测",
    "模块4 电化学性能预测",
    "总预测",
    "参数推荐"
])

with tab1:
    st.subheader("SEM 图像识别与量化分析")
    uploaded_file = st.file_uploader(
        "上传 SEM 图像", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="sem"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        result, binary, overlay, eq_diameters = analyze_sem_array(img_gray)

        c1, c2, c3 = st.columns(3)
        c1.image(img_gray, caption="原始灰度图", clamp=True)
        c2.image(binary, caption="阈值分割图", clamp=True)
        c3.image(overlay, caption="轮廓标注图", channels="BGR")

        st.write("### 量化结果")
        st.json(result)

        if len(eq_diameters) > 0:
            fig, ax = plt.subplots()
            ax.hist(eq_diameters, bins=20)
            ax.set_title("粒径分布")
            ax.set_xlabel("等效粒径")
            ax.set_ylabel("数量")
            st.pyplot(fig)

with tab2:
    st.subheader("Ag 纳米颗粒电沉积预测")

    col1, col2 = st.columns(2)

    with col1:
        current = st.number_input("施加电流大小 (mA)", value=10.0)
        ton = st.number_input("脉冲导通时间 (s)", value=0.2)
        toff = st.number_input("脉冲关断时间 (s)", value=0.5)
        cycles = st.number_input("循环次数", value=100)

    with col2:
        agno3 = st.number_input("AgNO3 浓度 (M)", value=0.02)
        kno3 = st.number_input("KNO3 浓度 (M)", value=0.10)
        pvp = st.number_input("PVP 浓度 (g/L)", value=1.0)
        citrate = st.number_input("柠檬酸钠浓度 (M)", value=0.01)

    if st.button("预测电沉积结果"):
        dep_result = predict_deposition({
            "current": current, "ton": ton, "toff": toff, "cycles": cycles,
            "agno3": agno3, "kno3": kno3, "pvp": pvp, "citrate": citrate
        })
        st.json(dep_result)

with tab3:
    st.subheader("PTFE 疏水层磁控溅射预测")

    col1, col2 = st.columns(2)
    with col1:
        power = st.number_input("溅射功率 (W)", value=60.0)
        pressure = st.number_input("保护气体压强 (Pa)", value=0.5)
    with col2:
        sputter_time = st.number_input("溅射时间 (s)", value=180.0)
        distance = st.number_input("靶材与基底距离 (cm)", value=5.0)

    if st.button("预测PTFE结果"):
        sputter_result = predict_sputter({
            "power": power,
            "pressure": pressure,
            "sputter_time": sputter_time,
            "distance": distance
        })
        st.json(sputter_result)

with tab4:
    st.subheader("电化学性能预测")

    col1, col2 = st.columns(2)

    with col1:
        particle_size = st.number_input("Ag 粒径", value=40.0)
        uniformity = st.number_input("Ag 均匀度", value=70.0)
        dispersion = st.number_input("Ag 分散度", value=60.0)
        overgrowth = st.number_input("Ag 过度生长程度", value=15.0)

    with col2:
        ptfe_thickness = st.number_input("PTFE 厚度", value=25.0)
        ptfe_uniformity = st.number_input("PTFE 均匀度", value=70.0)
        current_density = st.number_input("电流密度 (mA/cm²)", value=200.0)
        electrolyte_conc = st.number_input("电解液浓度 (M)", value=1.0)

    electrolyte_type = st.selectbox("电解液类型", ["K2CO3", "KHCO3", "KOH", "H2O"])

    if st.button("预测电化学性能"):
        elec_result = predict_electrochem({
            "particle_size": particle_size,
            "uniformity": uniformity,
            "dispersion": dispersion,
            "overgrowth": overgrowth,
            "ptfe_thickness": ptfe_thickness,
            "ptfe_uniformity": ptfe_uniformity,
            "current_density": current_density,
            "electrolyte_type": electrolyte_type,
            "electrolyte_conc": electrolyte_conc
        })
        st.json(elec_result)

with tab5:
    st.subheader("总模块：从工艺参数直接预测最终性能")
    st.write("自动串联模块2、模块3、模块4")

    if st.button("执行总预测"):
        dep_result = predict_deposition({
            "current": current, "ton": ton, "toff": toff, "cycles": cycles,
            "agno3": agno3, "kno3": kno3, "pvp": pvp, "citrate": citrate
        })

        sputter_result = predict_sputter({
            "power": power,
            "pressure": pressure,
            "sputter_time": sputter_time,
            "distance": distance
        })

        elec_result = predict_electrochem({
            "particle_size": dep_result["particle_size"],
            "uniformity": dep_result["uniformity"],
            "dispersion": dep_result["dispersion"],
            "overgrowth": dep_result["overgrowth"],
            "ptfe_thickness": sputter_result["ptfe_thickness"],
            "ptfe_uniformity": sputter_result["ptfe_uniformity"],
            "current_density": current_density,
            "electrolyte_type": electrolyte_type,
            "electrolyte_conc": electrolyte_conc
        })

        st.write("### 电沉积预测结果")
        st.json(dep_result)

        st.write("### PTFE 溅射预测结果")
        st.json(sputter_result)

        st.write("### 最终电化学性能预测")
        st.json(elec_result)

with tab6:
    st.subheader("下一组实验参数推荐")
    if st.button("生成推荐参数"):
        df = recommend_next_experiments(top_k=5)
        st.dataframe(df, use_container_width=True)
'''

files["README.txt"] = r'''
运行顺序：
1) python src/generate_simulated_data.py
2) python src/train_deposition.py
3) python src/train_sputter.py
4) python src/train_electrochem.py
5) streamlit run app.py

后续替换真实数据：
- 把真实 csv 放进 data/raw/
- 把 SEM 图像放进 data/raw/sem_images/
- 再按需要修改训练脚本输入路径
'''

for rel_path, content in files.items():
    file_path = BASE / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")

print("项目文件已自动生成完成。")
print(f"项目路径：{BASE}")