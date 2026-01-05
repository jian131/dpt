"""
Feature extraction: HSV Color Histogram + LBP Texture (TỰ CODE)
"""
import numpy as np
import cv2

# ============ HSV COLOR HISTOGRAM ============

def hsv_quantize(hsv_img, bins_H, bins_S, bins_V):
    """Quantize HSV image to bin indices"""
    H, W, _ = hsv_img.shape
    h = hsv_img[:, :, 0].astype(np.int32)  # [0,179]
    s = hsv_img[:, :, 1].astype(np.int32)  # [0,255]
    v = hsv_img[:, :, 2].astype(np.int32)  # [0,255]

    bin_h = (h * bins_H // 180).clip(0, bins_H - 1)
    bin_s = (s * bins_S // 256).clip(0, bins_S - 1)
    bin_v = (v * bins_V // 256).clip(0, bins_V - 1)

    return bin_h * (bins_S * bins_V) + bin_s * bins_V + bin_v

def compute_hsv_hist(img_bgr, bins_H, bins_S, bins_V):
    """Compute HSV histogram for entire image"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    idx = hsv_quantize(hsv, bins_H, bins_S, bins_V)
    K = bins_H * bins_S * bins_V
    hist = np.bincount(idx.ravel(), minlength=K).astype(np.float32)
    return hist / (hist.sum() + 1e-12)

def compute_grid_hsv_hist(img_bgr, grid, bins_H, bins_S, bins_V):
    """Compute HSV histogram with spatial grid"""
    gx, gy = grid
    H, W, _ = img_bgr.shape
    cell_h, cell_w = H // gy, W // gx
    K = bins_H * bins_S * bins_V

    hists = []
    for i in range(gy):
        for j in range(gx):
            y1, y2 = i * cell_h, (i + 1) * cell_h if i < gy - 1 else H
            x1, x2 = j * cell_w, (j + 1) * cell_w if j < gx - 1 else W
            cell = img_bgr[y1:y2, x1:x2]
            hists.append(compute_hsv_hist(cell, bins_H, bins_S, bins_V))

    return np.concatenate(hists)

# ============ LBP TEXTURE ============

def compute_lbp_hist(gray_img):
    """Compute LBP histogram (basic 3x3)"""
    H, W = gray_img.shape
    lbp = np.zeros((H - 2, W - 2), dtype=np.uint8)

    # 8 neighbors in order: TL, T, TR, R, BR, B, BL, L
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            center = gray_img[y, x]
            code = 0
            for k, (dy, dx) in enumerate(neighbors):
                if gray_img[y + dy, x + dx] >= center:
                    code |= (1 << k)
            lbp[y - 1, x - 1] = code

    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    return hist / (hist.sum() + 1e-12)

# ============ FEATURE EXTRACTION ============

def extract_feature(img_path, config, use_color=True, use_lbp=True):
    """Extract combined features from image"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")

    img = cv2.resize(img, config.image_size)
    features = []

    if use_color:
        color_vec = compute_grid_hsv_hist(
            img, config.grid, config.bins_H, config.bins_S, config.bins_V
        )
        features.append(color_vec * config.w_color)

    if use_lbp:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_vec = compute_lbp_hist(gray)
        features.append(lbp_vec * config.w_lbp)

    if not features:
        raise ValueError("Must enable at least one feature")

    vec = np.concatenate(features)
    vec = vec / (np.linalg.norm(vec) + 1e-12)  # L2 normalize
    return vec.astype(np.float32)
