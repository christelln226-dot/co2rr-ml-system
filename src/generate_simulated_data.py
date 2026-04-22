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
