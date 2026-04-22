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
