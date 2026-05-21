import os
import pandas as pd
from pathlib import Path

# SEM 图片总目录
base_dir = Path("data/raw/sem_images")

# 支持的图片格式
image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

rows = []

for sample_folder in sorted(base_dir.iterdir()):
    if not sample_folder.is_dir():
        continue

    sample_id = sample_folder.name  # ED001, ED002 ...

    # 只扫描一次，并用 suffix.lower() 判断格式，避免大小写重复
    image_files = [
        p for p in sample_folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    image_files = sorted(image_files)

    for idx, image_path in enumerate(image_files, start=1):
        image_id = image_path.stem  # ED001_01

        parts = image_id.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            view_no = int(parts[-1])
        else:
            view_no = idx

        rows.append({
            "图像编号": image_id,
            "样品编号": sample_id,
            "图像路径": str(image_path).replace("\\", "/"),
            "倍率": "",
            "标尺实际长度_nm": "",
            "标尺像素_px": "",
            "nm_per_pixel": "",
            "加速电压_kV": "",
            "工作距离_mm": "",
            "视野编号": view_no,
            "图像质量": "合格",
            "是否用于建模": "是",
            "备注": ""
        })

df = pd.DataFrame(rows)

output_path = "data/raw/sem_image_metadata_auto.xlsx"
df.to_excel(output_path, index=False)

print(f"已生成 SEM 图像信息表：{output_path}")
print(f"共扫描到 {len(df)} 张 SEM 图片")

# 顺便统计每个样品有几张图
count_df = df.groupby("样品编号").size().reset_index(name="图片数量")
print(count_df.to_string(index=False))