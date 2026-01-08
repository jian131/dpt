# CBIR Project - Team Guide 📚

**Đề tài:** Content-Based Image Retrieval (CBIR) + LSH Indexing
**Nhóm:** 4 thành viên
**Dataset:** Fashion-MNIST (500 ảnh, 10 classes)
**Repo:** https://github.com/jian131/dpt

---

## 📋 Tổng quan hệ thống

### **Luồng hoạt động chính:**

```
1. BUILD PHASE (Offline - làm 1 lần)
   Dataset (500 ảnh)
      ↓
   Extract Features (HSV + LBP) → features.npy (500 × 2560)
      ↓
   Build LSH Index → lsh_index.pkl (8 tables)
      ↓
   Save metadata → meta.csv

2. SEARCH PHASE (Online - real-time)
   Query Image
      ↓
   Extract Features → query vector (2560 dim)
      ↓
   LSH Query → candidates (50-100 ảnh)
      ↓
   Compute Distance (Chi²) → distances
      ↓
   Sort & Return Top-K → results

3. EVALUATION PHASE
   Test với 50 queries
      ↓
   Compute Precision@K, Recall@K
      ↓
   Compare Linear vs LSH (speedup)
```

### **Cấu trúc code:**

```
CBIR/
├── config.py          (41 lines)  - Cấu hình toàn bộ project
├── features.py        (97 lines)  - Extract HSV + LBP features
├── lsh.py             (83 lines)  - LSH indexing
├── build.py           (93 lines)  - Build features + index
├── search.py         (129 lines)  - Search với distance metrics
├── eval.py           (110 lines)  - Evaluation
├── gui.py            (228 lines)  - GUI demo (optional)
└── requirements.txt               - Dependencies
```

**Tổng code chính: 553 dòng** (không kể GUI)

---

## 👥 PHÂN CÔNG 4 THÀNH VIÊN

### 👤 **THÀNH VIÊN 1: Feature Extraction (HSV Color)**

**File:** `features.py` - Part 1 (HSV histogram)
**Dòng code:** ~60 lines

**Nhiệm vụ:**

- Implement HSV Color Histogram với spatial grid 3×3
- Quantization: H=16, S=4, V=4 bins
- Output: 2304-dim vector (9 cells × 256 bins)

**Báo cáo:**

- Giải thích tại sao dùng HSV thay vì RGB
- Demo histogram visualization
- So sánh features giữa 2 classes

---

### 👤 **THÀNH VIÊN 2: Feature Extraction (LBP Texture)**

**File:** `features.py` - Part 2 (LBP)
**Dòng code:** ~40 lines

**Nhiệm vụ:**

- Implement LBP (Local Binary Pattern) 3×3 basic
- 8 neighbors encoding → 256 patterns
- Output: 256-dim histogram

**Báo cáo:**

- Giải thích LBP encoding (binary pattern)
- Demo texture patterns khác nhau
- So sánh áo len vs áo lụa

---

### 👤 **THÀNH VIÊN 3: LSH Indexing**

**File:** `lsh.py` (83 lines)

**Nhiệm vụ:**

- Implement LSH với random hyperplanes
- Hash function: binary signature (12 bits)
- Multi-table (8 tables) để tăng recall
- Build & query index

**Báo cáo:**

- Giải thích LSH theory (collision probability)
- Demo số lượng candidates: 80/500
- Complexity: O(k) vs O(n)

---

### 👤 **THÀNH VIÊN 4: Search, Distance Metrics & Evaluation**

**Files:** `search.py` (129 lines) + `eval.py` (110 lines) + `build.py` (93 lines)

**Nhiệm vụ:**

- Implement 3 distance metrics: Chi², L1, L2
- Linear search vs LSH search
- Build pipeline: dataset → features → index
- Evaluation: Precision@K, Recall@K, Speedup

**Báo cáo:**

- So sánh 3 metrics (Chi² tốt nhất)
- Speedup: Linear 28ms vs LSH 1.5ms → 19x
- Precision/Recall curves

---

## 📊 PHÂN BỔ WORKLOAD

| Thành viên      | Code (lines) | Độ khó   | Tasks                              |
| --------------- | ------------ | -------- | ---------------------------------- |
| 1 - HSV         | 60           | ⭐⭐     | HSV quantization + Grid histogram  |
| 2 - LBP         | 40           | ⭐⭐     | LBP encoding + Histogram           |
| 3 - LSH         | 83           | ⭐⭐⭐⭐ | Random planes + Hash + Multi-table |
| 4 - Search/Eval | 332          | ⭐⭐⭐   | 3 Metrics + Search + Eval + Build  |

**Total: 515 lines thuật toán core**

---

## 👤 THÀNH VIÊN 1: HSV Color Histogram

### **Nhiệm vụ chi tiết:**

Implement thuật toán HSV Color Histogram với spatial grid

### **File phụ trách:** `features.py` (lines 8-50)

---

#### **1.1. HSV Color Histogram**

**Tại sao dùng HSV thay vì RGB?**

- **RGB:** Bị ảnh hưởng bởi ánh sáng (sáng/tối khác nhau)
- **HSV:** Tách màu sắc (H), độ bão hòa (S), độ sáng (V) → ổn định hơn

**Code chi tiết:**

```python
def hsv_quantize(hsv_img, bins_H, bins_S, bins_V):
    """
    Chuyển ảnh HSV thành bin indices

    Input:
        hsv_img: (H, W, 3) - ảnh HSV
        bins_H=16, bins_S=4, bins_V=4

    Output:
        idx: (H, W) - mỗi pixel → 1 số (0-255)
    """
    H, W, _ = hsv_img.shape

    # Tách 3 channels
    h = hsv_img[:, :, 0].astype(np.int32)  # [0,179]
    s = hsv_img[:, :, 1].astype(np.int32)  # [0,255]
    v = hsv_img[:, :, 2].astype(np.int32)  # [0,255]

    # Quantization: Chia khoảng thành bins
    # VD: H=90 → bin = 90*16/180 = 8
    bin_h = (h * bins_H // 180).clip(0, bins_H - 1)
    bin_s = (s * bins_S // 256).clip(0, bins_S - 1)
    bin_v = (v * bins_V // 256).clip(0, bins_V - 1)

    # Kết hợp 3 bins thành 1 index
    # Index = h*(S*V) + s*V + v
    # VD: (8,2,3) → 8*(4*4) + 2*4 + 3 = 139
    return bin_h * (bins_S * bins_V) + bin_s * bins_V + bin_v
```

**Giải thích:**

- Chia mỗi channel thành bins: H→16, S→4, V→4
- Total bins = 16×4×4 = **256 bins**
- Mỗi pixel thuộc 1 bin → tạo histogram

```python
def compute_grid_hsv_hist(img_bgr, grid, bins_H, bins_S, bins_V):
    """
    Compute spatial histogram (grid 3×3)

    Tại sao dùng grid?
    - Toàn bộ ảnh: Mất thông tin vị trí
    - Grid 3×3: Giữ thông tin "trên/dưới/trái/phải"

    Output: 9 cells × 256 bins = 2304 dim
    """
    gx, gy = grid  # (3, 3)
    H, W, _ = img_bgr.shape
    cell_h, cell_w = H // gy, W // gx
    K = bins_H * bins_S * bins_V  # 256

    hists = []
    for i in range(gy):  # 0,1,2
        for j in range(gx):  # 0,1,2
            # Cắt ảnh thành 9 cells
            y1, y2 = i * cell_h, (i + 1) * cell_h if i < gy - 1 else H
            x1, x2 = j * cell_w, (j + 1) * cell_w if j < gx - 1 else W
            cell = img_bgr[y1:y2, x1:x2]

            # Compute histogram cho cell này
            hists.append(compute_hsv_hist(cell, bins_H, bins_S, bins_V))

    return np.concatenate(hists)  # [2304]
```

**Ví dụ cụ thể:**

```
Ảnh áo đỏ (256×256):
┌─────┬─────┬─────┐
│ Đỏ  │ Đỏ  │ Đỏ  │  ← Top row: màu đỏ dominant
├─────┼─────┼─────┤
│Trắng│Trắng│Trắng│  ← Middle: màu trắng
├─────┼─────┼─────┤
│ Đỏ  │ Đỏ  │ Đỏ  │  ← Bottom: màu đỏ
└─────┴─────┴─────┘

→ 9 histograms riêng biệt
→ Phân biệt được "áo đỏ viền trắng" vs "áo trắng viền đỏ"
```

**Demo cho Thành viên 1:**

**1. Visualize HSV histogram:**

```python
import matplotlib.pyplot as plt

img_path = "dataset/T-shirt/0.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))

# Compute histogram cho 1 cell
hsv = cv2.cvtColor(img[:85, :85], cv2.COLOR_BGR2HSV)
idx = hsv_quantize(hsv, 16, 4, 4)
hist = np.bincount(idx.ravel(), minlength=256)

# Plot
plt.bar(range(256), hist)
plt.title("HSV Histogram - Top-left cell")
plt.xlabel("Bin")
plt.ylabel("Frequency")
plt.show()
```

**2. So sánh 2 classes:**

```python
# T-shirt (xám) vs Dress (trắng)
t_shirt = compute_grid_hsv_hist(cv2.imread("dataset/T-shirt/0.jpg"), (3,3), 16,4,4)
dress = compute_grid_hsv_hist(cv2.imread("dataset/Dress/0.jpg"), (3,3), 16,4,4)

# Cosine similarity
sim = np.dot(t_shirt, dress) / (np.linalg.norm(t_shirt) * np.linalg.norm(dress))
print(f"Similarity: {sim:.3f}")  # Low (~0.3-0.4)
```

**3. Spatial information:**

```python
# So sánh global vs spatial
global_hist = compute_hsv_hist(img, 16, 4, 4)  # 256 dim
spatial_hist = compute_grid_hsv_hist(img, (3,3), 16,4,4)  # 2304 dim

print(f"Global: {global_hist.shape}")    # (256,)
print(f"Spatial: {spatial_hist.shape}")  # (2304,)
# Spatial giữ được thông tin vị trí màu sắc!
```

---

## 👤 THÀNH VIÊN 2: LBP Texture

### **Nhiệm vụ chi tiết:**

Implement Local Binary Pattern để capture texture

### **File phụ trách:** `features.py` (lines 51-76)

---

#### **2.1. LBP Theory**

**Code chi tiết:**

```python
def compute_lbp_hist(gray_img):
    """
    Local Binary Pattern - Mã hóa texture

    Cách hoạt động:
    1. Lấy 1 pixel làm center
    2. So sánh với 8 neighbors xung quanh
    3. Tạo binary code (8 bits) → 1 số (0-255)

    Input: gray_img (H, W) - ảnh grayscale
    Output: histogram (256,) - phân bố LBP codes
    """
    H, W = gray_img.shape
    lbp = np.zeros((H - 2, W - 2), dtype=np.uint8)

    # 8 hướng: ↖ ↑ ↗ → ↘ ↓ ↙ ←
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,1),
                 (1,1), (1,0), (1,-1), (0,-1)]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            center = gray_img[y, x]  # Pixel trung tâm
            code = 0

            # So sánh với 8 neighbors
            for k, (dy, dx) in enumerate(neighbors):
                neighbor = gray_img[y + dy, x + dx]

                # Nếu neighbor >= center → bit = 1
                if neighbor >= center:
                    code |= (1 << k)  # Set bit thứ k

            lbp[y - 1, x - 1] = code  # Code từ 0-255

    # Tạo histogram
    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    return hist / (hist.sum() + 1e-12)  # Normalize
```

**Ví dụ cụ thể:**

```
       50  60  70
       40 [55] 65   ← Center pixel = 55
       30  45  75

So sánh:
↖ 50 < 55 → 0
↑ 60 > 55 → 1
↗ 70 > 55 → 1
→ 65 > 55 → 1
↘ 75 > 55 → 1
↓ 45 < 55 → 0
↙ 30 < 55 → 0
← 40 < 55 → 0

Binary: 01111000 → Decimal: 120
→ LBP code = 120
```

**Ý nghĩa:**

- Code 120 xuất hiện nhiều → Texture có pattern cụ thể
- Mỗi texture khác nhau → Histogram khác nhau
- Ví dụ:
  - Áo len: Nhiều codes 11111111, 00000000 (thô ráp)
  - Áo lụa: Nhiều codes 01010101 (mịn màng)

**Demo cho Thành viên 2:**

**1. Visualize LBP codes:**

```python
img = cv2.imread("dataset/Coat/0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Compute LBP
lbp_hist = compute_lbp_hist(img)

# Plot histogram
plt.bar(range(256), lbp_hist)
plt.title("LBP Histogram - Coat texture")
plt.xlabel("LBP Code (0-255)")
plt.ylabel("Frequency")
plt.show()
```

**2. So sánh textures:**

```python
# Áo len (thô) vs Áo lụa (mịn)
coat_gray = cv2.imread("dataset/Coat/0.jpg", cv2.IMREAD_GRAYSCALE)
dress_gray = cv2.imread("dataset/Dress/0.jpg", cv2.IMREAD_GRAYSCALE)

coat_lbp = compute_lbp_hist(coat_gray)
dress_lbp = compute_lbp_hist(dress_gray)

# Compare
plt.subplot(1,2,1)
plt.bar(range(256), coat_lbp)
plt.title("Coat (rough)")

plt.subplot(1,2,2)
plt.bar(range(256), dress_lbp)
plt.title("Dress (smooth)")
plt.show()
```

**3. LBP image visualization:**

```python
# Visualize LBP values
H, W = img.shape
lbp_img = np.zeros((H-2, W-2), dtype=np.uint8)
# ... (compute LBP for each pixel)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(lbp_img, cmap='gray')
plt.title("LBP codes")
plt.show()
```

---

## 👤 THÀNH VIÊN 3: LSH Indexing

### **Nhiệm vụ chi tiết:**

Implement Locality-Sensitive Hashing với random hyperplanes

### **File phụ trách:** `lsh.py` (83 lines)

    Output: (2560,) = 2304 (HSV) + 256 (LBP)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")

    img = cv2.resize(img, config.image_size)  # 256×256
    features = []

    if use_color:
        # HSV histogram: 2304 dim
        color_vec = compute_grid_hsv_hist(
            img, config.grid, config.bins_H, config.bins_S, config.bins_V
        )
        features.append(color_vec * config.w_color)  # Weight: 0.6

    if use_lbp:
        # LBP texture: 256 dim
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_vec = compute_lbp_hist(gray)
        features.append(lbp_vec * config.w_lbp)  # Weight: 0.4

    if not features:
        raise ValueError("Must enable at least one feature")

    # Kết hợp
    vec = np.concatenate(features)  # [2560]

    # L2 normalization: Đưa về unit vector
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    return vec.astype(np.float32)

````

**Tại sao normalize?**

- Đảm bảo tất cả vectors có độ dài = 1
- Distance metrics fair hơn (không bị ảnh hưởng bởi magnitude)

---

### **Demo cho Thành viên 1:**

**1. Visualize histogram:**

```python
import matplotlib.pyplot as plt

img_path = "dataset/T-shirt/0.jpg"
feat = extract_feature(img_path, config)

# Plot HSV histogram
plt.subplot(1,2,1)
plt.bar(range(256), feat[:256])
plt.title("HSV Histogram (Cell 1)")

# Plot LBP histogram
plt.subplot(1,2,2)
plt.bar(range(256), feat[2304:])
plt.title("LBP Histogram")
plt.show()
````

**2. So sánh features:**

```python
# T-shirt vs Trouser
t_shirt = extract_feature("dataset/T-shirt/0.jpg", config)
trouser = extract_feature("dataset/Trouser/0.jpg", config)

# Cosine similarity
sim = np.dot(t_shirt, trouser)
print(f"Similarity: {sim:.3f}")  # ~0.3-0.4 (khác nhau)

# T-shirt vs T-shirt khác
t_shirt2 = extract_feature("dataset/T-shirt/1.jpg", config)
sim2 = np.dot(t_shirt, t_shirt2)
print(f"Similarity: {sim2:.3f}")  # ~0.7-0.9 (giống nhau)
```

---

## 👤 THÀNH VIÊN 2: LSH Indexing

### **Nhiệm vụ:**

Implement Locality-Sensitive Hashing để tăng tốc search:

- Random hyperplanes projection
- Multi-table hashing
- Query candidates retrieval

### **File phụ trách:** `lsh.py`

---

#### **2.1. Lý thuyết LSH**

**Vấn đề:**

- Database có 500 ảnh
- Linear search: So sánh query với **500 ảnh** → O(n)
- Nếu n lớn (1M ảnh) → quá chậm!

**Giải pháp: LSH**

- Chia không gian thành "buckets" (nhóm)
- Ảnh giống nhau → cùng bucket (high probability)
- Search chỉ trong bucket → O(k) với k << n

**Cách hoạt động:**

```
Không gian 2560 chiều
        ↓
Random Hyperplanes (12 planes)
        ↓
Binary Hash Code (12 bits)
        ↓
Bucket ID (0-4095)
        ↓
Hash Table (dict)
```

**Ví dụ 2D:**

```
       |
   ●   |   ○
   ● ● | ○ ○
───────+────────
   ●   |   ○
       |

Plane chia không gian thành 2 phần:
- Trái: ● (similar items)
- Phải: ○ (similar items)

Với 2 planes → 2² = 4 buckets
Với 12 planes → 2¹² = 4096 buckets
```

---

#### **2.2. Code chi tiết**

```python
class LSHIndex:
    def __init__(self, num_tables, num_planes, dim, seed=42):
        """
        LSH Index với random hyperplanes

        Args:
            num_tables: Số lượng hash tables (8)
            num_planes: Số planes mỗi table (12)
            dim: Feature dimension (2560)
            seed: Random seed
        """
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.dim = dim
        self.seed = seed

        # Generate random planes
        self.planes = self._make_planes()

        # Hash tables (8 tables)
        self.tables = [dict() for _ in range(num_tables)]

        self.num_vectors = 0
```

**Tại sao 8 tables?**

- 1 table có thể miss một số ảnh tương tự
- 8 tables → 8 lần random → tăng recall
- Trade-off: Nhiều tables → chậm hơn

---

#### **2.3. Random Hyperplanes**

```python
def _make_planes(self):
    """
    Tạo random hyperplanes

    Hyperplane: ax₁ + bx₂ + ... + c·x₂₅₆₀ = 0
    Represented by vector (a, b, ..., c)

    Output: 8 tables × 12 planes = 96 vectors (12, 2560)
    """
    np.random.seed(self.seed)
    planes = []

    for _ in range(self.num_tables):
        # Random normal distribution
        p = np.random.randn(self.num_planes, self.dim).astype(np.float32)

        # Normalize to unit vectors
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)

        planes.append(p)

    return planes
```

**Tại sao normalize?**

- Đảm bảo chỉ quan tâm đến **direction**, không phải magnitude
- Dot product = cosine similarity

---

#### **2.4. Hash Function**

```python
def _hash(self, vec, planes):
    """
    Hash vector thành binary code

    Steps:
    1. Dot product với 12 planes
    2. Nếu > 0 → bit = 1, else → bit = 0
    3. Kết hợp 12 bits thành 1 số (0-4095)

    Args:
        vec: (2560,) - feature vector
        planes: (12, 2560) - random hyperplanes

    Returns:
        hash_val: 0-4095 (12-bit number)
    """
    # Dot product với tất cả planes
    dots = np.dot(planes, vec)  # (12,)

    # Threshold tại 0
    bits = (dots >= 0).astype(np.uint8)  # [1,0,1,1,0,...]

    # Convert binary to decimal
    hash_val = 0
    for i, bit in enumerate(bits):
        if bit:
            hash_val |= (1 << i)  # Set bit thứ i

    return hash_val
```

**Ví dụ cụ thể:**

```python
vec = [0.1, 0.5, -0.3, ..., 0.2]  # 2560 dim

planes = [
    [0.2, 0.1, ...],  # plane 1
    [-0.1, 0.3, ...], # plane 2
    ...
]

dots = [0.15, -0.05, 0.23, ...]  # 12 values

bits:
plane 1:  0.15 > 0 → 1
plane 2: -0.05 < 0 → 0
plane 3:  0.23 > 0 → 1
...
→ [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]

Binary: 101101001101
Decimal: 2893

→ hash_val = 2893
```

---

#### **2.5. Build Index**

```python
def fit(self, vectors):
    """
    Build hash tables từ database

    Args:
        vectors: (500, 2560) - tất cả features
    """
    self.num_vectors = len(vectors)
    self.tables = [dict() for _ in range(self.num_tables)]

    # Hash từng vector vào 8 tables
    for vid in range(len(vectors)):
        for tid in range(self.num_tables):
            # Hash vector này
            h = self._hash(vectors[vid], self.planes[tid])

            # Add vào bucket
            if h not in self.tables[tid]:
                self.tables[tid][h] = []
            self.tables[tid][h].append(vid)
```

**Ví dụ sau khi build:**

```python
tables[0] = {
    2893: [0, 15, 234],     # Bucket 2893 có 3 ảnh
    1024: [1, 2, 88, 99],   # Bucket 1024 có 4 ảnh
    ...
}

tables[1] = {
    567: [0, 10, 20],
    ...
}
```

**Observation:**

- Ảnh giống nhau (similar features) → same hash code
- Rơi vào cùng bucket!

---

#### **2.6. Query**

```python
def query(self, vec):
    """
    Tìm candidates cho query vector

    Steps:
    1. Hash query vào 8 tables
    2. Lấy union tất cả buckets
    3. Return candidates

    Returns:
        set of image IDs (50-100 candidates)
    """
    candidates = set()

    # Query từng table
    for tid in range(self.num_tables):
        # Hash query
        h = self._hash(vec, self.planes[tid])

        # Lấy bucket này
        if h in self.tables[tid]:
            candidates.update(self.tables[tid][h])

    return candidates
```

**Ví dụ:**

```python
query_vec = [...]  # T-shirt features

# Hash vào 8 tables
table 0: hash = 2893 → bucket có [0, 15, 234]
table 1: hash = 567  → bucket có [0, 10, 20]
table 2: hash = 3012 → bucket có [15, 99]
...

# Union tất cả
candidates = {0, 10, 15, 20, 99, 234, ...}
→ ~80 candidates (thay vì 500!)
```

**Speedup:**

- Linear: So sánh với 500 ảnh
- LSH: So sánh với 80 ảnh → **6x nhanh hơn**
- Thực tế: 19-22x (vì LSH query cũng nhanh)

---

### **Demo cho Thành viên 3:**

**1. Collision probability test:**

```python
# Tạo 2 vectors tương tự 90%
v1 = np.random.randn(2560)
v2 = 0.9 * v1 + 0.1 * np.random.randn(2560)
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# Build index với v1
index = LSHIndex(8, 12, 2560, seed=42)
index.fit(np.array([v1]))

# Query với v2
candidates = index.query(v2)
print(f"v2 → v1 collision: {0 in candidates}")  # True với high probability

# Test với vector random (không giống)
v3 = np.random.randn(2560)
v3 = v3 / np.linalg.norm(v3)
candidates3 = index.query(v3)
print(f"v3 → v1 collision: {0 in candidates3}")  # False
```

**2. Candidates reduction:**

```python
# Load features
features = np.load("artifacts/features.npy")  # (500, 2560)

# Build index
index = LSHIndex(8, 12, 2560)
index.fit(features)

# Query nhiều ảnh
num_candidates = []
for i in range(50):
    cands = index.query(features[i])
    num_candidates.append(len(cands))

print(f"Avg candidates: {np.mean(num_candidates):.0f}/{len(features)}")  # ~80/500
print(f"Reduction: {len(features) / np.mean(num_candidates):.1f}x")      # ~6.3x
```

**3. Hash distribution:**

```python
# Phân bố hash codes trong 1 table
hash_counts = {}
for i in range(len(features)):
    h = index._hash(features[i], index.planes[0])
    hash_counts[h] = hash_counts.get(h, 0) + 1

# Plot distribution
import matplotlib.pyplot as plt
plt.hist(hash_counts.values(), bins=20)
plt.xlabel("Bucket size")
plt.ylabel("Frequency")
plt.title("Hash distribution (Table 0)")
plt.show()
```

---

## 👤 THÀNH VIÊN 4: Search, Distance Metrics & Evaluation

### **Nhiệm vụ chi tiết:**

Implement search pipeline, distance metrics, và evaluation metrics

### **Files phụ trách:**

- `search.py` (129 lines) - Distance metrics + Search algorithms
- `eval.py` (110 lines) - Evaluation metrics
- `build.py` (93 lines) - Build pipeline

**2. Số lượng candidates:**

```python
# Build index với 500 ảnh
features = np.load("artifacts/features.npy")
index = LSHIndex(8, 12, 2560)
index.fit(features)

# Query 50 ảnh
num_candidates = []
for i in range(50):
    cands = index.query(features[i])
    num_candidates.append(len(cands))

print(f"Avg candidates: {np.mean(num_candidates):.0f}")  # ~80
print(f"Reduction: {500 / np.mean(num_candidates):.1f}x")  # ~6x
```

---

## 👤 THÀNH VIÊN 3: Search & Distance Metrics

### **Nhiệm vụ:**

- Implement 3 distance metrics: Chi², L1, L2
- Linear search vs LSH search
- Build features từ dataset

### **File phụ trách:** `search.py`, `build.py`

---

#### **3.1. Distance Metrics**

**Tại sao cần distance?**

- Features là vectors (2560 dim)
- Cần đo "độ khác biệt" giữa 2 vectors
- Distance nhỏ → similar, distance lớn → different

---

##### **Chi-Square Distance**

```python
def chi2_distance(a, b, eps=1e-10):
    """
    Chi-square distance cho histograms

    Formula: χ² = 0.5 × Σ [(aᵢ - bᵢ)² / (aᵢ + bᵢ)]

    Tại sao dùng cho histogram?
    - Normalize by sum (aᵢ + bᵢ) → robust
    - Không bị ảnh hưởng bởi magnitude

    Args:
        a, b: (2560,) hoặc (N, 2560)

    Returns:
        distance: scalar hoặc (N,)
    """
    diff = a - b  # Element-wise difference
    sum_ab = a + b + eps  # Tránh chia 0

    # Chi² formula
    chi2 = 0.5 * np.sum((diff * diff) / sum_ab, axis=-1)

    return chi2
```

**Ví dụ:**

```python
# Histogram A: [0.5, 0.3, 0.2]
# Histogram B: [0.4, 0.4, 0.2]

diff = [0.1, -0.1, 0.0]
sum = [0.9, 0.7, 0.4]

chi2 = 0.5 * (0.1²/0.9 + 0.1²/0.7 + 0²/0.4)
     = 0.5 * (0.011 + 0.014 + 0)
     = 0.0125
```

**Ưu điểm:**

- Tốt cho histogram comparison
- Robust với outliers

---

##### **L1 Distance (Manhattan)**

```python
def l1_distance(a, b):
    """
    L1 (Manhattan) distance

    Formula: L1 = Σ |aᵢ - bᵢ|

    Ý nghĩa:
    - Tổng absolute differences
    - "Khoảng cách đi trên lưới ô"
    """
    return np.sum(np.abs(a - b), axis=-1)
```

**Ví dụ:**

```python
a = [1, 2, 3]
b = [2, 1, 4]

L1 = |1-2| + |2-1| + |3-4|
   = 1 + 1 + 1
   = 3
```

---

##### **L2 Distance (Euclidean)**

```python
def l2_distance(a, b):
    """
    L2 (Euclidean) distance

    Formula: L2 = √(Σ (aᵢ - bᵢ)²)

    Ý nghĩa:
    - "Khoảng cách đường thẳng"
    - Phổ biến nhất
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))
```

**Ví dụ:**

```python
a = [1, 2]
b = [4, 6]

L2 = √((1-4)² + (2-6)²)
   = √(9 + 16)
   = √25
   = 5
```

---

##### **So sánh 3 metrics:**

```python
a = np.array([0.5, 0.3, 0.2])
b = np.array([0.4, 0.4, 0.2])

print(f"Chi²: {chi2_distance(a, b):.4f}")  # 0.0125
print(f"L1:   {l1_distance(a, b):.4f}")    # 0.2
print(f"L2:   {l2_distance(a, b):.4f}")    # 0.1414
```

**Khi nào dùng metric nào?**

- **Chi²:** Histogram features (HSV, LBP) ✅ Tốt nhất cho CBIR
- **L1:** Simple, fast
- **L2:** General purpose

---

#### **3.2. Linear Search**

```python
def search_linear(query_vec, features, metric, k):
    """
    Linear search - brute force

    Steps:
    1. Compute distance với TẤT CẢ 500 ảnh
    2. Sort distances
    3. Return top-K smallest

    Complexity: O(n) với n=500
    """
    start = time.time()

    # Compute distances (vectorized)
    distances = pairwise_distance(query_vec, features, metric)
    # distances: (500,)

    # Get top-K indices
    topk_ids = topk_indices(distances, k)

    elapsed = (time.time() - start) * 1000  # ms

    return topk_ids, distances[topk_ids], len(features), elapsed
```

**Ví dụ:**

```python
query = features[0]  # T-shirt
distances = [0.000, 0.234, 0.156, ..., 0.892]  # 500 values

# Sort
sorted_ids = [0, 10, 25, 5, ...]  # Indices sorted by distance
topk_ids = sorted_ids[:10]  # Top-10
```

**Bottleneck:** Phải tính 500 distances → chậm!

---

#### **3.3. LSH Search**

```python
def search_lsh(query_vec, features, index, metric, k):
    """
    LSH search - fast

    Steps:
    1. Query LSH index → candidates (80 ảnh)
    2. Compute distance chỉ với candidates
    3. Sort & return top-K

    Complexity: O(k) với k≈80 << 500
    """
    start = time.time()

    # Query LSH index
    candidates = index.query(query_vec)  # ~80 IDs

    if not candidates:
        # Fallback to linear
        return search_linear(query_vec, features, metric, k)

    # Get candidate features
    cand_ids = list(candidates)
    cand_feats = features[cand_ids]  # (80, 2560)

    # Compute distances chỉ với candidates
    distances = pairwise_distance(query_vec, cand_feats, metric)

    # Top-K trong candidates
    local_topk = topk_indices(distances, min(k, len(distances)))

    # Map back to global IDs
    topk_ids = np.array([cand_ids[i] for i in local_topk])
    topk_dists = distances[local_topk]

    elapsed = (time.time() - start) * 1000

    return topk_ids, topk_dists, len(candidates), elapsed
```

**Ví dụ:**

```python
# Linear search
Compute 500 distances → 30ms

# LSH search
Query index → 0.5ms
Compute 80 distances → 5ms
Total → 5.5ms

Speedup = 30 / 5.5 ≈ 5.5x
```

**Thực tế speedup cao hơn (19-22x) vì:**

- LSH query rất nhanh (hash lookup)
- Vectorized operations với ít candidates

---

#### **3.4. Build Pipeline**

**File: `build.py`**

```python
def load_dataset(root_dir):
    """
    Scan dataset folder

    Structure:
    dataset/
      ├── T-shirt/0.jpg, 1.jpg, ...
      ├── Trouser/0.jpg, ...
      └── ...

    Returns:
        paths: ["dataset/T-shirt/0.jpg", ...]
        labels: [0, 0, 0, ..., 1, 1, ...]
        class_names: ["T-shirt", "Trouser", ...]
    """
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"Dataset not found: {root_dir}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    paths = []
    labels = []

    for label, class_dir in enumerate(class_dirs):
        imgs = sorted(class_dir.glob("*.jpg"))
        for img_path in imgs:
            paths.append(str(img_path))
            labels.append(label)

    return paths, labels, class_names
```

```python
def main():
    """Build features + LSH index"""

    # 1. Load dataset
    paths, labels, class_names = load_dataset(args.dataset)
    print(f"Found {len(paths)} images, {len(class_names)} classes")

    # 2. Extract features
    features = []
    for path in tqdm(paths, desc="Extracting"):
        feat = extract_feature(path, cfg)
        features.append(feat)

    features = np.array(features)  # (500, 2560)
    print(f"Shape: {features.shape}")

    # 3. Save features
    np.save(cfg.features_path, features)

    # 4. Save metadata
    meta_df = pd.DataFrame({
        'id': range(len(paths)),
        'path': paths,
        'label': labels,
        'class_name': [class_names[l] for l in labels]
    })
    meta_df.to_csv(cfg.meta_path, index=False)

    # 5. Build LSH index
    index = LSHIndex(cfg.num_tables, cfg.num_planes, features.shape[1], cfg.lsh_seed)
    index.fit(features)
    index.save(cfg.index_path)

    print("DONE!")
```

**Chạy:**

```bash
python build.py --dataset dataset
```

**Output:**

```
Found 500 images, 10 classes
Extracting: 100%|████| 500/500 [01:02<00:00,  7.97it/s]
Shape: (500, 2560)
Saved: artifacts/features.npy, artifacts/meta.csv
Saved: artifacts/lsh_index.pkl
DONE!
```

---

### **Demo cho Thành viên 3:**

**1. So sánh 3 metrics:**

```python
query = features[0]
database = features[1:]

chi2_dists = chi2_distance(query, database)
l1_dists = l1_distance(query, database)
l2_dists = l2_distance(query, database)

# Top-10 cho mỗi metric
print("Chi²:", topk_indices(chi2_dists, 10))
print("L1:  ", topk_indices(l1_dists, 10))
print("L2:  ", topk_indices(l2_dists, 10))

# Có thể khác nhau! → Chi² tốt nhất cho histogram
```

**2. Speedup chart:**

```python
import matplotlib.pyplot as plt

times_linear = []
times_lsh = []

for i in range(50):
    query = features[i]

    # Linear
    _, _, _, t_linear = search_linear(query, features, 'chi2', 10)
    times_linear.append(t_linear)

    # LSH
    _, _, _, t_lsh = search_lsh(query, features, index, 'chi2', 10)
    times_lsh.append(t_lsh)

plt.boxplot([times_linear, times_lsh], labels=['Linear', 'LSH'])
plt.ylabel('Time (ms)')
plt.title(f'Speedup: {np.mean(times_linear)/np.mean(times_lsh):.1f}x')
plt.show()
```

---

## 👤 THÀNH VIÊN 4: Evaluation + GUI

### **Nhiệm vụ:**

- Implement Precision@K, Recall@K
- Compare Linear vs LSH
- Build GUI demo với Tkinter

### **File phụ trách:** `eval.py`, `gui.py`

---

#### **4.1. Evaluation Metrics**

##### **Precision@K**

**Định nghĩa:**

```
Precision@K = (Số ảnh đúng trong top-K) / K
```

**Ví dụ:**

```python
Query: T-shirt (class 0)
Top-10 results:
#1: T-shirt ✅
#2: Shirt   ❌
#3: T-shirt ✅
#4: Coat    ❌
#5: T-shirt ✅
#6: Shirt   ❌
#7: T-shirt ✅
#8: Pullover❌
#9: T-shirt ✅
#10: T-shirt✅

Đúng: 6/10
Precision@10 = 6/10 = 0.60 = 60%
```

**Code:**

```python
def compute_precision(query_label, result_labels, k):
    """
    Precision@K

    Args:
        query_label: 0 (T-shirt)
        result_labels: [0, 5, 0, 3, 0, ...] top-K labels
        k: 10

    Returns:
        precision: 0.0-1.0
    """
    # Count correct predictions
    correct = sum([1 for label in result_labels[:k] if label == query_label])

    return correct / k
```

---

##### **Recall@K**

**Định nghĩa:**

```
Recall@K = (Số ảnh đúng trong top-K) / (Tổng số ảnh đúng trong database)
```

**Ví dụ:**

```python
Dataset có 50 T-shirts
Top-10 có 6 T-shirts

Recall@10 = 6 / 50 = 0.12 = 12%
```

**Code:**

```python
def compute_recall(query_label, result_labels, k, total_relevant):
    """
    Recall@K

    Args:
        query_label: 0
        result_labels: [0, 5, 0, ...] top-K
        k: 10
        total_relevant: 50 (total T-shirts)

    Returns:
        recall: 0.0-1.0
    """
    correct = sum([1 for label in result_labels[:k] if label == query_label])

    return correct / total_relevant
```

---

##### **Trade-off: Precision vs Recall**

```
K càng lớn:
- Precision giảm (nhiều ảnh sai hơn)
- Recall tăng (cover nhiều ảnh đúng hơn)

VD với T-shirt:
K=5:   Precision=80%, Recall=8%
K=10:  Precision=60%, Recall=12%
K=20:  Precision=40%, Recall=16%
```

---

#### **4.2. Evaluation Pipeline**

```python
def evaluate(features, meta_df, index, mode='linear', k=10, num_queries=50, metric='chi2'):
    """
    Evaluate search performance

    Args:
        features: (500, 2560)
        meta_df: DataFrame với class_name
        index: LSHIndex hoặc None
        mode: 'linear', 'lsh', 'both'
        k: Top-K
        num_queries: Số query test
        metric: 'chi2', 'l1', 'l2'

    Returns:
        results: dict với precision, recall, time
    """
    # Random sample queries
    np.random.seed(cfg.eval_seed)
    query_ids = np.random.choice(len(features), num_queries, replace=False)

    precisions = []
    recalls = []
    times = []

    for qid in tqdm(query_ids, desc=f"Evaluating {mode}"):
        query_vec = features[qid]
        query_label = meta_df.iloc[qid]['label']

        # Count total relevant
        total_relevant = (meta_df['label'] == query_label).sum()

        # Search
        if mode == 'linear':
            topk_ids, _, _, search_time = search_linear(query_vec, features, metric, k)
        else:  # lsh
            topk_ids, _, _, search_time = search_lsh(query_vec, features, index, metric, k)

        # Get labels
        result_labels = meta_df.iloc[topk_ids]['label'].values

        # Compute metrics
        prec = compute_precision(query_label, result_labels, k)
        rec = compute_recall(query_label, result_labels, k, total_relevant)

        precisions.append(prec)
        recalls.append(rec)
        times.append(search_time)

    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'time': np.mean(times)
    }
```

**Chạy evaluation:**

```bash
python eval.py --k 10 --num_queries 50 --mode both --metric chi2
```

**Output:**

```
LINEAR MODE
  Precision@10: 0.7200 (72%)
  Recall@10:    0.1440 (14.4%)
  Avg time:     28.5ms

LSH MODE
  Precision@10: 0.7200 (72%)  ← Same!
  Recall@10:    0.1440 (14.4%)
  Avg time:     1.5ms

SPEEDUP: 19.0x ⚡
```

**Observations:**

- Precision/Recall giống nhau → LSH không mất accuracy
- Time giảm 19x → LSH rất hiệu quả!

---

#### **4.3. GUI Implementation**

**File: `gui.py`**

```python
class CBIRGUI:
    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.root.title("CBIR - Content-Based Image Retrieval")
        self.root.geometry("1200x700")

        # Load artifacts
        self.load_artifacts()

        # Create UI
        self.create_widgets()

        self.query_image_path = None
```

**Load artifacts:**

```python
def load_artifacts(self):
    """Load features, metadata, LSH index"""
    try:
        # Load features (500, 2560)
        self.features = np.load(cfg.features_path)

        # Load metadata (CSV manually vì pandas chậm)
        self.meta = []
        with open(cfg.meta_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.meta.append(row)

        # Load LSH index
        self.lsh_index = LSHIndex.load(cfg.index_path)

        messagebox.showinfo("Success", f"Loaded {len(self.features)} images")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot load artifacts:\n{e}")
        self.root.quit()
```

**UI Layout:**

```
┌─────────────────────────────────────────────────────┐
│  Query Image     [Chọn Ảnh Query]    🔍 Search     │
│  ┌─────────┐                                        │
│  │         │     ○ LSH (Nhanh)     Top-K: [10]     │
│  │ Preview │     ○ Linear (Chính xác)              │
│  └─────────┘                                        │
│  ✓ Found 10 results in 1.74ms (LSH mode)           │
├─────────────────────────────────────────────────────┤
│ Results:                                            │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                    │
│ │ 1 │ │ 2 │ │ 3 │ │ 4 │ │ 5 │                    │
│ └───┘ └───┘ └───┘ └───┘ └───┘                    │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                    │
│ │ 6 │ │ 7 │ │ 8 │ │ 9 │ │10 │                    │
│ └───┘ └───┘ └───┘ └───┘ └───┘                    │
└─────────────────────────────────────────────────────┘
```

**Search function:**

```python
def search(self):
    """Perform search when button clicked"""
    if not self.query_image_path:
        messagebox.showwarning("Warning", "Please select query image!")
        return

    try:
        # Extract features
        query_feat = extract_feature(self.query_image_path, cfg)

        # Get settings
        mode = self.search_mode.get()  # 'lsh' or 'linear'
        topk = self.topk_var.get()     # 10

        # Search
        if mode == "lsh":
            indices, distances, num_cand, search_time = search_lsh(
                query_feat, self.features, self.lsh_index, 'chi2', topk
            )
        else:
            indices, distances, num_cand, search_time = search_linear(
                query_feat, self.features, 'chi2', topk
            )

        # Display results
        self.display_results(indices, distances, search_time, mode)

    except Exception as e:
        messagebox.showerror("Error", f"Search error:\n{e}")
```

**Display results:**

```python
def display_results(self, indices, distances, search_time, mode):
    """Show results in grid"""
    # Clear previous
    for widget in self.results_container.winfo_children():
        widget.destroy()

    # Update info
    self.info_label.config(
        text=f"✓ Found {len(indices)} results in {search_time:.2f}ms ({mode.upper()} mode)"
    )

    # Display in 5-column grid
    cols = 5
    for idx, (i, dist) in enumerate(zip(indices, distances)):
        row = idx // cols
        col = idx % cols

        # Frame for each result
        result_frame = tk.Frame(self.results_container, relief=tk.RAISED, bd=2)
        result_frame.grid(row=row, column=col, padx=5, pady=5)

        # Load & display image
        img_path = self.meta[i]['path']
        img = Image.open(img_path)
        img.thumbnail((150, 150))
        photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(result_frame, image=photo)
        img_label.image = photo  # Keep reference!
        img_label.pack()

        # Info
        class_name = self.meta[i]['class_name']
        tk.Label(result_frame, text=f"#{idx+1}: {class_name}",
                 font=("Arial", 9, "bold")).pack()
        tk.Label(result_frame, text=f"Distance: {dist:.3f}",
                 font=("Arial", 8), fg="gray").pack()
```

**Chạy GUI:**

```bash
python gui.py
```

---

### **Demo cho Thành viên 4:**

**1. Precision/Recall curves:**

```python
import matplotlib.pyplot as plt

ks = [1, 3, 5, 10, 20, 30, 50]
precisions = []
recalls = []

for k in ks:
    results = evaluate(features, meta_df, index, 'lsh', k, 50, 'chi2')
    precisions.append(results['precision'])
    recalls.append(results['recall'])

plt.plot(recalls, precisions, 'o-')
plt.xlabel('Recall@K')
plt.ylabel('Precision@K')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
```

**2. Compare metrics:**

```python
metrics = ['chi2', 'l1', 'l2']
results = {}

for metric in metrics:
    r = evaluate(features, meta_df, index, 'lsh', 10, 50, metric)
    results[metric] = r['precision']

plt.bar(metrics, [results[m] for m in metrics])
plt.ylabel('Precision@10')
plt.title('Metric Comparison')
plt.show()
```

---

## 📊 Tổng kết luồng hoạt động

### **Phase 1: BUILD (Offline)**

```
1. Load Dataset (build.py)
   dataset/T-shirt/*.jpg → paths, labels

2. Extract Features (Thành viên 1 + 2)
   - HSV Color Histogram (Thành viên 1): 2304-dim
   - LBP Texture (Thành viên 2): 256-dim
   → Combine = 2560-dim vector

3. Build LSH Index (Thành viên 3)
   500 vectors → 8 hash tables

4. Save Artifacts (Thành viên 4)
   features.npy, meta.csv, lsh_index.pkl
```

### **Phase 2: SEARCH (Online)**

```
1. Query Image Input
   Chọn query ảnh từ dataset

2. Extract Features (Thành viên 1 + 2)
   - HSV histogram (Thành viên 1)
   - LBP histogram (Thành viên 2)
   → Query vector 2560-dim

3. LSH Query (Thành viên 3)
   Hash query → ~80 candidates

4. Compute Distance (Thành viên 4)
   Chi² distance với candidates

5. Sort & Return Top-K (Thành viên 4)
   Top-10 smallest distances
```

### **Phase 3: EVALUATION (Offline)**

```
1. Random Sample Queries (Thành viên 4)
   50 query images

2. Run Search (Thành viên 4)
   - Linear search: 500 vectors
   - LSH search: ~80 candidates

3. Compute Metrics (Thành viên 4)
   Precision@K, Recall@K, Time

4. Compare Performance (Thành viên 4)
   Speedup: Linear vs LSH
```

---

## 🎯 Câu hỏi thường gặp

### **Q1: Tại sao dùng Fashion-MNIST thay vì CIFAR-10?**

A: Fashion-MNIST nhẹ hơn (30MB vs 170MB), download nhanh, và vẫn đủ thú vị cho demo.

### **Q2: Tại sao grid 3×3 cho HSV?**

A: Trade-off giữa spatial information và feature dimension:

- 1×1 (global): Mất thông tin vị trí
- 3×3: Đủ capture "trên/dưới", dimension không quá lớn
- 5×5: Quá chi tiết, dimension explode

### **Q3: Tại sao 8 tables, 12 planes?**

A: Tuning based on experiments:

- 4 tables: Miss nhiều candidates
- 8 tables: Balance tốt
- 16 tables: Chậm hơn, không cải thiện nhiều

### **Q4: LSH có luôn nhanh hơn Linear?**

A: Không! Nếu n nhỏ (< 100 ảnh), Linear có thể nhanh hơn vì LSH có overhead.

### **Q5: Làm sao improve accuracy?**

A:

1. Thêm features: SIFT, HOG, deep features (CNN)
2. Tăng grid: 3×3 → 4×4
3. Feature fusion: Combine multiple features
4. Re-ranking: Spatial verification

---

## 📝 Checklist trước báo cáo

### **Tất cả thành viên:**

- [ ] Code đã push lên GitHub
- [ ] Comment đầy đủ trong code
- [ ] Hiểu rõ code của mình (giải thích từng dòng)
- [ ] Test code: `python build.py`, `python search.py`, `python eval.py`

### **Thành viên 1 - HSV Color (~60 lines):**

- [ ] Demo histogram visualization cho 3 classes
- [ ] So sánh spatial vs global histogram
- [ ] Giải thích quantization: Tại sao 16×4×4 bins?
- [ ] Trả lời: HSV tốt hơn RGB như thế nào?

### **Thành viên 2 - LBP Texture (~40 lines):**

- [ ] Demo LBP codes visualization
- [ ] So sánh texture: Coat (rough) vs Dress (smooth)
- [ ] Giải thích 8-neighbor encoding
- [ ] Trả lời: LBP capture texture pattern ra sao?

### **Thành viên 3 - LSH Indexing (83 lines):**

- [ ] Demo collision probability test
- [ ] Chart candidates reduction: 500 → 80
- [ ] Giải thích random hyperplanes
- [ ] Trả lời: Tại sao 8 tables? Tại sao 12 planes?

### **Thành viên 4 - Search/Eval/Build (332 lines):**

- [ ] So sánh 3 metrics: Chi² vs L1 vs L2
- [ ] Speedup chart: Linear vs LSH (~19x)
- [ ] Precision-Recall curve cho K khác nhau
- [ ] Trả lời: Trade-off giữa accuracy và speed?

**Lưu ý:** File `gui.py` (228 lines) là **optional bonus** cho demo trực quan, **không bắt buộc** trong phân công!

---

## 🚀 Mở rộng trong tương lai

1. **Deep Features:** Dùng CNN (ResNet, VGG) thay vì HSV+LBP
2. **Re-ranking:** Spatial verification, query expansion
3. **Scalability:** Test với 1M ảnh
4. **Web App:** Deploy lên Flask/FastAPI
5. **Mobile App:** Android/iOS với TensorFlow Lite

---

**Good luck! 💪**
