import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =========================
# 文件路径设置
# =========================
META_PATH = "data/raw/sem_image_metadata_auto.xlsx"
SHEET_NAME = "02_SEM图像信息"

OUT_SINGLE = "data/processed/sem_single_results.xlsx"
OUT_SUMMARY = "data/processed/sem_sample_summary.xlsx"


# =========================
# 基础函数
# =========================
def ensure_parent_dir(file_path):
    """
    确保输出文件所在文件夹存在。
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def read_image_unicode(image_path):
    """
    兼容中文路径、tif/png/jpg 等图像读取。
    OpenCV 直接 imread 有时不支持中文路径，所以用 np.fromfile + cv2.imdecode。
    """
    if not os.path.exists(image_path):
        return None

    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def crop_sem_content(img):
    """
    裁掉 SEM 图底部黑色参数栏，只保留有效形貌区域。

    你的 SEM 图底部有：
    - 日期
    - HV
    - mag
    - WD
    - scale bar
    这些不应该参与颗粒识别。

    当前采用固定裁剪比例：保留上方 86%。
    后续如果发现仍然把底部黑条识别进去，可以把 0.86 改小一点，比如 0.82。
    """
    h, w = img.shape[:2]
    crop_h = int(h * 0.86)
    return img[:crop_h, :]


# =========================
# SEM 图像分析核心函数
# =========================
def analyze_sem_image(image_path, nm_per_pixel):
    """
    对单张 SEM 图进行传统图像处理分析。

    输出指标：
    - 覆盖率
    - 颗粒数量
    - 颗粒密度
    - 平均粒径
    - 粒径标准差
    - 均匀度
    - 过度生长指数
    - 团聚指数
    """

    img = read_image_unicode(image_path)

    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    # 裁掉底部标尺和参数栏
    img_crop = crop_sem_content(img)

    # 转灰度
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    # 高斯滤波降噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 阈值分割
    # 亮区域通常对应沉积颗粒/团聚体
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 形态学开运算去除小噪声
    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # 查找轮廓
    contours, _ = cv2.findContours(
        binary_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    total_area_px = h * w

    particle_areas_px = []
    particle_diameters_px = []

    # 面积筛选参数
    # min_area 过小会把噪点当颗粒
    # max_area 过大会把大片背景/过曝区域纳入
    min_area_px = 5
    max_area_px = total_area_px * 0.05

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area_px:
            continue

        if area > max_area_px:
            continue

        particle_areas_px.append(area)

        # 等效圆直径，单位 px
        diameter_px = 2 * np.sqrt(area / np.pi)
        particle_diameters_px.append(diameter_px)

    particle_count = len(particle_diameters_px)

    if particle_count > 0:
        particle_areas_px = np.array(particle_areas_px)
        particle_diameters_px = np.array(particle_diameters_px)

        mean_size_px = float(np.mean(particle_diameters_px))
        median_size_px = float(np.median(particle_diameters_px))
        size_std_px = float(np.std(particle_diameters_px))

        mean_size_nm = mean_size_px * nm_per_pixel
        median_size_nm = median_size_px * nm_per_pixel
        size_std_nm = size_std_px * nm_per_pixel

        coverage_pct = float(np.sum(particle_areas_px) / total_area_px * 100)

        # 粒径变异系数 CV，越小越均匀
        cv_size = size_std_px / mean_size_px if mean_size_px > 0 else np.nan

        # 均匀度定义为 100*(1-CV)，最低不低于 0
        uniformity = float(max(0, 100 * (1 - cv_size)))

        # 过度生长指数：大于 2 倍中位粒径的颗粒比例
        overgrowth_index = float(
            np.mean(particle_diameters_px > 2 * median_size_px)
        )

        # 团聚指数：大颗粒面积占总颗粒面积比例
        large_mask = particle_diameters_px > 1.5 * median_size_px

        if np.sum(particle_areas_px) > 0:
            agglomeration_index = float(
                np.sum(particle_areas_px[large_mask]) / np.sum(particle_areas_px)
            )
        else:
            agglomeration_index = 0.0

    else:
        mean_size_px = np.nan
        median_size_px = np.nan
        size_std_px = np.nan

        mean_size_nm = np.nan
        median_size_nm = np.nan
        size_std_nm = np.nan

        coverage_pct = 0.0
        uniformity = np.nan
        overgrowth_index = np.nan
        agglomeration_index = np.nan

    # 面积换算：px² → μm²
    # nm_per_pixel² = 每像素面积 nm²
    area_um2 = total_area_px * (nm_per_pixel ** 2) / 1_000_000

    if area_um2 > 0:
        particle_density_per_um2 = particle_count / area_um2
    else:
        particle_density_per_um2 = np.nan

    return {
        "覆盖率_pct": coverage_pct,
        "颗粒数量": particle_count,
        "颗粒密度_个每um2": particle_density_per_um2,
        "平均粒径_px": mean_size_px,
        "平均粒径_nm": mean_size_nm,
        "中位粒径_px": median_size_px,
        "中位粒径_nm": median_size_nm,
        "粒径标准差_px": size_std_px,
        "粒径标准差_nm": size_std_nm,
        "均匀度": uniformity,
        "过度生长指数": overgrowth_index,
        "团聚指数": agglomeration_index,
    }


# =========================
# 读取 Excel
# =========================
def load_metadata():
    """
    读取 02_SEM图像信息 工作表。

    你的 Excel 表格结构是：
    第 1 行：标题
    第 2 行：说明
    第 3 行：空行/格式行
    第 4 行：真正表头

    所以 header=3。
    """
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"找不到 SEM 图像信息表：{META_PATH}\n"
            f"请确认 Excel 文件已经放到 data/raw/ 下，并命名为 sem_image_metadata_auto.xlsx"
        )

    df = pd.read_excel(
        META_PATH,
        sheet_name=SHEET_NAME,
        header=3
    )

    # 去除全空行
    df = df.dropna(how="all")

    # 去掉列名前后空格
    df.columns = [str(c).strip() for c in df.columns]

    return df


# =========================
# 主程序
# =========================
def main():
    print("正在读取 SEM 图像信息表...")
    df = load_metadata()

    print("读取到的列名：")
    print(list(df.columns))

    required_cols = [
        "图像编号",
        "样品编号",
        "图像路径",
        "倍率",
        "标尺实际长度_nm",
        "标尺像素_px",
        "换算比例_nm_per_pixel",
        "图像质量",
        "是否用于建模",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"缺少必要列：{col}\n"
                f"当前读取到的列名为：{list(df.columns)}\n"
                f"请检查 Excel 的 02_SEM图像信息 表头是否正确。"
            )

    rows = []

    print("\n开始批量分析 SEM 图像...")

    for _, row in df.iterrows():
        image_id = str(row["图像编号"]).strip()
        sample_id = str(row["样品编号"]).strip()
        image_path = str(row["图像路径"]).strip()

        image_quality = str(row.get("图像质量", "合格")).strip()
        use_for_model = str(row.get("是否用于建模", "是")).strip()

        # 跳过空白行
        if image_id == "" or image_id.lower() == "nan":
            continue

        # 只分析合格图像
        if image_quality != "合格":
            print(f"跳过非合格图像：{image_id}")
            continue

        # 只分析用于建模的图像
        if use_for_model != "是":
            print(f"跳过非建模图像：{image_id}")
            continue

        if not os.path.exists(image_path):
            print(f"图像不存在，跳过：{image_id} -> {image_path}")
            continue

        nm_per_pixel = row["换算比例_nm_per_pixel"]

        if pd.isna(nm_per_pixel):
            print(f"缺少换算比例，跳过：{image_id}")
            continue

        try:
            nm_per_pixel = float(nm_per_pixel)
        except Exception:
            print(f"换算比例不是数字，跳过：{image_id}")
            continue

        if nm_per_pixel <= 0:
            print(f"换算比例小于等于 0，跳过：{image_id}")
            continue

        try:
            features = analyze_sem_image(image_path, nm_per_pixel)

            result = {
                "图像编号": image_id,
                "样品编号": sample_id,
                "图像路径": image_path,
                "倍率": row["倍率"],
                "标尺实际长度_nm": row["标尺实际长度_nm"],
                "标尺像素_px": row["标尺像素_px"],
                "换算比例_nm_per_pixel": nm_per_pixel,
                "图像质量": image_quality,
                "是否用于建模": use_for_model,
            }

            result.update(features)
            rows.append(result)

            print(f"完成分析：{image_id}")

        except Exception as e:
            print(f"分析失败：{image_id}，原因：{e}")

    result_df = pd.DataFrame(rows)

    ensure_parent_dir(OUT_SINGLE)
    result_df.to_excel(OUT_SINGLE, index=False)

    print("\n==============================")
    print(f"单图分析结果已保存：{OUT_SINGLE}")
    print(f"共完成 {len(result_df)} 张图像分析")
    print("==============================")

    if result_df.empty:
        print("没有生成有效分析结果。请检查：")
        print("1. 是否用于建模 是否为 是")
        print("2. 图像质量 是否为 合格")
        print("3. 图像路径 是否正确")
        print("4. 换算比例_nm_per_pixel 是否为有效数字")
        return

    # =========================
    # 样品级汇总
    # =========================
    numeric_cols = [
        "覆盖率_pct",
        "颗粒数量",
        "颗粒密度_个每um2",
        "平均粒径_nm",
        "中位粒径_nm",
        "粒径标准差_nm",
        "均匀度",
        "过度生长指数",
        "团聚指数",
    ]

    summary_mean = (
        result_df
        .groupby("样品编号")[numeric_cols]
        .mean()
        .reset_index()
    )

    summary_std = (
        result_df
        .groupby("样品编号")[numeric_cols]
        .std()
        .reset_index()
    )

    summary_std = summary_std.rename(
        columns={col: col + "_标准差" for col in numeric_cols}
    )

    summary_df = summary_mean.merge(
        summary_std,
        on="样品编号",
        how="left"
    )

    count_df = (
        result_df
        .groupby("样品编号")
        .size()
        .reset_index(name="参与分析图像数")
    )

    summary_df = summary_df.merge(
        count_df,
        on="样品编号",
        how="left"
    )

    summary_df.to_excel(OUT_SUMMARY, index=False)

    print(f"样品汇总结果已保存：{OUT_SUMMARY}")


if __name__ == "__main__":
    main()