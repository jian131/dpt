"""
HSV Color Histogram (TỰ CODE)
- hsv_quantize: chuyển HSV image thành index map
- compute_hsv_hist: tính histogram cho 1 ảnh
- compute_grid_hsv_hist: tính histogram theo grid (spatial color)

Input: BGR image từ OpenCV
Output: histogram vector đã normalize
"""
import numpy as np
import cv2


def hsv_quantize(hsv_img, bins_H, bins_S, bins_V):
    """
    Chuyển HSV image thành index map (quantization).

    Args:
        hsv_img: ndarray shape (H, W, 3), dtype uint8
                 H in [0, 179], S in [0, 255], V in [0, 255]
        bins_H: số bins cho H
        bins_S: số bins cho S
        bins_V: số bins cho V

    Returns:
        idx_map: ndarray shape (H, W), dtype int
                 index = binH * (bins_S * bins_V) + binS * bins_V + binV
    """
    H, W, _ = hsv_img.shape
    h_channel = hsv_img[:, :, 0].astype(np.int32)  # [0, 179]
    s_channel = hsv_img[:, :, 1].astype(np.int32)  # [0, 255]
    v_channel = hsv_img[:, :, 2].astype(np.int32)  # [0, 255]

    # Quantize theo bins
    bin_h = (h_channel * bins_H // 180).clip(0, bins_H - 1)
    bin_s = (s_channel * bins_S // 256).clip(0, bins_S - 1)
    bin_v = (v_channel * bins_V // 256).clip(0, bins_V - 1)

    # Tính index tuyến tính
    idx_map = bin_h * (bins_S * bins_V) + bin_s * bins_V + bin_v

    return idx_map


def compute_hsv_hist(image_bgr, bins_H, bins_S, bins_V):
    """
    Tính HSV histogram cho toàn bộ ảnh.

    Args:
        image_bgr: BGR image từ cv2.imread, shape (H, W, 3)
        bins_H, bins_S, bins_V: số bins

    Returns:
        hist: ndarray shape (K,), K = bins_H * bins_S * bins_V
              đã normalize (sum = 1)
    """
    # Chuyển sang HSV
    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Quantize
    idx_map = hsv_quantize(hsv_img, bins_H, bins_S, bins_V)

    # Tính histogram bằng np.bincount (nhanh)
    K = bins_H * bins_S * bins_V
    hist = np.bincount(idx_map.ravel(), minlength=K).astype(np.float32)

    # Normalize
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist


def compute_grid_hsv_hist(image_bgr, grid, bins_H, bins_S, bins_V):
    """
    Tính HSV histogram theo grid (spatial color histogram).
    Chia ảnh thành grid (gx, gy) ô, mỗi ô tính histogram riêng rồi concat.

    Args:
        image_bgr: BGR image, shape (H, W, 3)
        grid: tuple (gx, gy), số ô theo chiều ngang và dọc
        bins_H, bins_S, bins_V: số bins

    Returns:
        vec: ndarray shape (gx * gy * K,), K = bins_H * bins_S * bins_V
             mỗi ô đã normalize riêng
    """
    gx, gy = grid
    H, W, _ = image_bgr.shape

    # Kích thước mỗi ô
    cell_h = H // gy
    cell_w = W // gx

    K = bins_H * bins_S * bins_V
    all_hists = []

    # Duyệt qua từng ô
    for i in range(gy):
        for j in range(gx):
            # Crop ô (i, j)
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < gy - 1 else H
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < gx - 1 else W

            cell_img = image_bgr[y_start:y_end, x_start:x_end]

            # Tính histogram cho ô này
            cell_hist = compute_hsv_hist(cell_img, bins_H, bins_S, bins_V)
            all_hists.append(cell_hist)

    # Concat tất cả histogram
    vec = np.concatenate(all_hists)

    return vec
