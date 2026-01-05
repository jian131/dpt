# Content-Based Image Retrieval (CBIR) + LSH Indexing

Hệ thống tìm kiếm ảnh tương tự (query-by-image) với LSH indexing.

## 📌 Tính năng

- ✅ **HSV Color Histogram** (grid 3x3) - TỰ CODE
- ✅ **LBP Texture** (3x3) - TỰ CODE
- ✅ **LSH Indexing** (random hyperplanes) - TỰ CODE
- ✅ **Distance metrics**: L1, L2, Chi-square - TỰ CODE
- ✅ **Evaluation**: Precision@K, Recall@K
- ✅ **Speedup**: 19-22x nhanh hơn Linear search

## 🚀 Hướng dẫn cho thành viên nhóm

### 1. Clone repo từ GitHub

```bash
git clone https://github.com/jian131/dpt.git
cd dpt
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Cài packages

```bash
pip install -r requirements.txt
```

### 4. Tải dataset + artifacts

**Lưu ý:** Dataset (~8MB) và artifacts (~4MB) KHÔNG có trên GitHub.

**👉 Nhận từ leader nhóm** qua:

- Google Drive / OneDrive / WeTransfer
- Giải nén vào thư mục gốc project

Cấu trúc sau khi giải nén:

```
dpt/
├── dataset/          ← Folder này
│   ├── red/*.jpg
│   ├── blue/*.jpg
│   └── ...
├── artifacts/        ← Folder này
│   ├── features.npy
│   ├── meta.csv
│   └── lsh_index.pkl
└── ...
```

### 5. Test chạy

```bash
# Search ảnh
python search.py --query "dataset/red/0.jpg" --mode lsh --topk 10

# Evaluation
python eval.py --k 10 --num_queries 30 --mode both
```

## 📁 Cấu trúc code

```
dpt/
├── config.py        (41 dòng) - Cấu hình
├── features.py      (76 dòng) - HSV + LBP (TỰ CODE)
├── lsh.py           (73 dòng) - LSH indexing (TỰ CODE)
├── build.py         (76 dòng) - Build features + index
├── search.py       (106 dòng) - Search + distance (TỰ CODE)
├── eval.py         (110 dòng) - Evaluation
├── requirements.txt
└── README.md
```

**Tổng: 482 dòng code**

## 🎯 Kết quả

| Metric           | Linear  | LSH        |
| ---------------- | ------- | ---------- |
| **Precision@10** | 100%    | 100%       |
| **Recall@10**    | 34.48%  | 34.48%     |
| **Query time**   | 6.81 ms | 0.36 ms    |
| **Speedup**      | 1x      | **19x** ⚡ |

## 💡 Nếu muốn build lại từ đầu

Nếu có dataset mới (ảnh trong `dataset/<class>/*.jpg`):

```bash
python build.py --dataset dataset
```

Sẽ tạo:

- `artifacts/features.npy` - Feature vectors (N × 2560)
- `artifacts/meta.csv` - Metadata
- `artifacts/lsh_index.pkl` - LSH index

## 📦 Chia sẻ dataset/artifacts với nhóm

**Leader nhóm làm:**

1. Nén dataset + artifacts:

```bash
# Windows PowerShell
Compress-Archive -Path dataset,artifacts -DestinationPath cbir-data.zip

# Linux/Mac
zip -r cbir-data.zip dataset artifacts
```

2. Upload lên Google Drive / OneDrive

3. Chia sẻ link cho thành viên

**Thành viên nhận:**

- Download `cbir-data.zip`
- Giải nén vào folder project
- Chạy `python search.py --query ...`

## 🛠️ Requirements

- Python 3.8+
- OpenCV, NumPy, Pandas, Matplotlib, tqdm

## 👥 Nhóm

Bài tập lớn CBIR - 2026

---

**Repo:** https://github.com/jian131/dpt
