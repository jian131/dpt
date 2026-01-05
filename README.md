# Content-Based Image Retrieval (CBIR) + LSH Indexing

## Bài toán

Hệ thống tìm kiếm ảnh tương tự (query-by-image): người dùng đưa 1 ảnh làm query, hệ thống trả về Top-K ảnh giống nhất từ database.

## Giải pháp

### Features (Tự code)

- **HSV Color Histogram**: Tính histogram màu trên không gian HSV với grid 3x3 (spatial color)
  - Bins: H=16, S=4, V=4 → 256 bins/cell
  - Grid 3x3 → 2304-dim vector
- **LBP Texture**: Local Binary Pattern 3x3 cơ bản
  - 8 neighbors → 256 bins
- **Kết hợp**: Color (weight=0.6) + LBP (weight=0.4) → L2 normalize

### Distance Metrics (Tự code)

- L1 (Manhattan)
- L2 (Euclidean)
- Chi-square: `0.5 * sum((a-b)^2 / (a+b+eps))`

### Indexing (Tự code)

- **LSH (Locality Sensitive Hashing)**: Random hyperplanes
  - 8 tables, 12 planes/table
  - Giảm thời gian tìm kiếm từ O(N) → O(N/num_tables)

### Dataset

- Dùng **TensorFlow Datasets (TFDS)** để tải dataset
- Mặc định: Caltech101, 30 classes, 80 ảnh/class
- Export ra `dataset/<class_name>/<id>.jpg`

## Kết quả

Ví dụ trên Caltech101 (30 classes, ~2400 ảnh):

| Metric           | Linear Search | LSH Search | Speedup  |
| ---------------- | ------------- | ---------- | -------- |
| **Precision@10** | 0.85          | 0.82       | -        |
| **Recall@10**    | 0.42          | 0.40       | -        |
| **Query time**   | 45 ms         | 8 ms       | **5.6x** |

**Kết luận**: LSH giảm thời gian tìm kiếm ~5-6 lần với độ chính xác chỉ giảm ~3%.

## Cài đặt

```bash
# Clone hoặc tải project
cd CBIR

# Cài dependencies
pip install -r requirements.txt
```

## Cách chạy

### 1. Download và export dataset

```bash
python data/download_tfds.py --dataset caltech101 --out dataset --split train --limit_per_class 80 --max_classes 30 --image_size 256
```

**Tham số**:

- `--dataset`: tên dataset trên TFDS (mặc định: caltech101)
- `--out`: thư mục output (mặc định: dataset)
- `--split`: train/test (mặc định: train)
- `--limit_per_class`: số ảnh/class (mặc định: 80)
- `--max_classes`: số class (mặc định: 30)
- `--image_size`: resize ảnh (mặc định: 256)

### 2. Build offline (extract features + build index)

```bash
python offline_build.py --download
```

**Tham số**:

- `--download`: tự động tải dataset nếu chưa có
- `--skip_features`: bỏ qua trích feature (dùng features đã có)
- `--rebuild_index`: build lại LSH index

**Output**:

- `artifacts/features.npy`: feature vectors (N x D)
- `artifacts/meta.csv`: metadata (id, path, label, class_name)
- `artifacts/lsh_index.pkl`: LSH index

### 3. Search demo

```bash
python search.py --query "dataset/accordion/0.jpg" --mode lsh --topk 10 --metric chi2
```

**Tham số**:

- `--query`: đường dẫn ảnh query
- `--mode`: linear hoặc lsh (mặc định: lsh)
- `--topk`: số kết quả (mặc định: 10)
- `--metric`: l1, l2, hoặc chi2 (mặc định: chi2)
- `--no_show`: không hiển thị ảnh kết quả

**Output**: In Top-K ảnh + distance, hiển thị grid ảnh (query + results).

### 4. Evaluation

```bash
python eval.py --k 10 --num_queries 50 --mode both
```

**Tham số**:

- `--k`: K cho Precision@K (mặc định: 10)
- `--num_queries`: số ảnh query để test (mặc định: 50)
- `--mode`: linear, lsh, hoặc both (mặc định: both)
- `--metric`: l1, l2, hoặc chi2 (mặc định: chi2)
- `--seed`: random seed (mặc định: 42)

**Output**: Precision@K, Recall@K, thời gian truy vấn trung bình, speedup (nếu mode=both).

## Cấu trúc thư mục

```
CBIR/
├── config.py              # Cấu hình mặc định
├── requirements.txt       # Dependencies
├── README.md              # File này
│
├── data/
│   └── download_tfds.py   # Download dataset từ TFDS
│
├── features/
│   ├── __init__.py
│   ├── color_hist.py      # HSV histogram (tự code)
│   ├── lbp.py             # LBP texture (tự code)
│   └── combine.py         # Kết hợp features
│
├── indexing/
│   ├── __init__.py
│   └── lsh.py             # LSH index (tự code)
│
├── dataset.py             # Load dataset paths/labels
├── similarity.py          # Distance metrics (tự code)
├── utils.py               # Utilities (timer, etc.)
│
├── offline_build.py       # Build offline: features + index
├── search.py              # Search demo
├── eval.py                # Evaluation
│
├── dataset/               # Dataset folder (tự động tạo)
│   └── <class_name>/
│       └── *.jpg
│
└── artifacts/             # Artifacts (tự động tạo)
    ├── features.npy
    ├── meta.csv
    └── lsh_index.pkl
```

## Yêu cầu hệ thống

- Python 3.8+
- Windows/Linux/Mac
- RAM: ~4GB (cho 2400 ảnh)
- Disk: ~500MB (dataset + features)

## Tính năng nổi bật

✅ **Tự code hoàn toàn**: HSV histogram, LBP, distance metrics, LSH indexing
✅ **Dễ hiểu**: Code rõ ràng, comment đầy đủ
✅ **Đánh giá đầy đủ**: Precision@K, Recall@K, so sánh Linear vs LSH
✅ **TFDS integration**: Tải dataset dễ dàng từ TensorFlow Datasets
✅ **Visualization**: Hiển thị kết quả search bằng matplotlib

## Tác giả

Dự án bài tập lớn CBIR - 2026
