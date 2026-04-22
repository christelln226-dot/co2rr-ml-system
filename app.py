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
