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
