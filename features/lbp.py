"""
LBP (Local Binary Pattern) texture descriptor (TỰ CODE)
- Basic 3x3 LBP: so sánh 8 neighbors với center pixel
- Trả về histogram 256 bins

Input: Grayscale image (uint8)
Output: LBP histogram đã normalize
"""
import numpy as np


def compute_lbp_hist(gray_img):
    """
    Tính LBP histogram cho ảnh grayscale.

    LBP basic 3x3:
    - Lấy 8 neighbors xung quanh center pixel theo thứ tự cố định
    - Neighbors: (y-1,x-1), (y-1,x), (y-1,x+1), (y,x+1),
                 (y+1,x+1), (y+1,x), (y+1,x-1), (y,x-1)
    - Bit k = 1 nếu neighbor[k] >= center else 0
    - Label = sum(bit_k << k) => giá trị 0..255

    Args:
        gray_img: ndarray shape (H, W), dtype uint8

    Returns:
        hist: ndarray shape (256,), đã normalize
    """
    H, W = gray_img.shape

    # Tạo label map (chỉ tính cho vùng trong, bỏ border 1 pixel)
    lbp_labels = np.zeros((H - 2, W - 2), dtype=np.uint8)

    # Định nghĩa 8 neighbors theo thứ tự (dy, dx)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),  # top-left, top, top-right
        (0, 1),                       # right
        (1, 1), (1, 0), (1, -1),      # bottom-right, bottom, bottom-left
        (0, -1)                       # left
    ]

    # Duyệt qua từng pixel (bỏ border)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            center = gray_img[y, x]
            label = 0

            # So sánh 8 neighbors
            for k, (dy, dx) in enumerate(neighbors):
                neighbor_val = gray_img[y + dy, x + dx]
                if neighbor_val >= center:
                    label |= (1 << k)

            lbp_labels[y - 1, x - 1] = label

    # Tính histogram
    hist = np.bincount(lbp_labels.ravel(), minlength=256).astype(np.float32)

    # Normalize
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist
