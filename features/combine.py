"""
Combine features: HSV histogram + LBP
- extract_feature: đọc ảnh và trích đặc trưng
- Hỗ trợ ablation: chỉ color, chỉ LBP, hoặc cả hai
"""
import cv2
import numpy as np
import sys
import os

# Import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.color_hist import compute_grid_hsv_hist
from features.lbp import compute_lbp_hist
import config as cfg


def extract_feature(image_path, use_color=True, use_lbp=True):
    """
    Trích đặc trưng từ ảnh.

    Args:
        image_path: đường dẫn ảnh
        use_color: có dùng color histogram không
        use_lbp: có dùng LBP không

    Returns:
        feature_vec: ndarray float32, đã L2 normalize

    Note:
        - Color: grid HSV histogram với weight w_color
        - LBP: LBP histogram với weight w_lbp
        - Cuối cùng L2 normalize toàn bộ vector
    """
    # Đọc ảnh
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Resize về kích thước chuẩn
    img_bgr = cv2.resize(img_bgr, cfg.image_size)

    features = []

    # 1. Color histogram
    if use_color:
        color_vec = compute_grid_hsv_hist(
            img_bgr,
            grid=cfg.grid,
            bins_H=cfg.bins_H,
            bins_S=cfg.bins_S,
            bins_V=cfg.bins_V
        )
        # Áp dụng weight
        color_vec = color_vec * cfg.w_color
        features.append(color_vec)

    # 2. LBP texture
    if use_lbp:
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lbp_vec = compute_lbp_hist(gray_img)
        # Áp dụng weight
        lbp_vec = lbp_vec * cfg.w_lbp
        features.append(lbp_vec)

    # Kiểm tra có ít nhất 1 feature
    if len(features) == 0:
        raise ValueError("Phải bật ít nhất 1 feature: use_color hoặc use_lbp")

    # Concat tất cả features
    feature_vec = np.concatenate(features)

    # L2 normalize toàn bộ vector
    norm = np.linalg.norm(feature_vec)
    if norm > 1e-12:
        feature_vec = feature_vec / norm

    return feature_vec.astype(np.float32)
