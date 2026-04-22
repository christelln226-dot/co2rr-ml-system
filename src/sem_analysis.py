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
