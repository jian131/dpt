"""
Offline build: tải dataset, trích features, build LSH index.

Steps:
    1. Download dataset (nếu cần)
    2. Load image paths và labels
    3. Extract features cho tất cả ảnh
    4. Save features + metadata
    5. Build LSH index
    6. Save index

Usage:
    python offline_build.py --download
    python offline_build.py --skip_features --rebuild_index
"""
import argparse
import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config as cfg
from dataset import load_paths_labels
from features import extract_feature
from indexing import LSHIndex
from utils import mkdir_if_not_exists, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Offline build: extract features + build index")
    parser.add_argument("--download", action="store_true", help="Download dataset nếu chưa có")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction (dùng features đã có)")
    parser.add_argument("--rebuild_index", action="store_true", help="Rebuild LSH index")
    parser.add_argument("--use_color", action="store_true", default=True, help="Use color histogram")
    parser.add_argument("--use_lbp", action="store_true", default=True, help="Use LBP texture")
    return parser.parse_args()


def download_dataset():
    """
    Gọi download_tfds.py để tải và export dataset.
    """
    print("\n" + "="*60)
    print("STEP 1: DOWNLOAD DATASET")
    print("="*60)

    dataset_dir = Path(cfg.out_dir_dataset)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"[INFO] Dataset folder '{dataset_dir}' đã tồn tại, skip download.")
        return

    print("[INFO] Đang download dataset...")
    cmd = [
        sys.executable,
        "data/download_tfds.py",
        "--dataset", cfg.dataset_name,
        "--out", cfg.out_dir_dataset,
        "--split", cfg.export_split,
        "--limit_per_class", str(cfg.limit_per_class),
        "--max_classes", str(cfg.max_classes),
        "--image_size", str(cfg.image_size[0])
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] Download dataset thất bại!")
        sys.exit(1)

    print("[INFO] Download dataset hoàn tất!")


def extract_all_features(paths, use_color=True, use_lbp=True):
    """
    Trích features cho tất cả ảnh.

    Args:
        paths: list of str, đường dẫn ảnh
        use_color: dùng color histogram
        use_lbp: dùng LBP

    Returns:
        features: ndarray shape (N, D)
    """
    print("\n" + "="*60)
    print("STEP 2: EXTRACT FEATURES")
    print("="*60)
    print(f"[INFO] Số ảnh: {len(paths)}")
    print(f"[INFO] Features: color={use_color}, lbp={use_lbp}")

    features_list = []
    start_time = time.time()

    for path in tqdm(paths, desc="Extracting"):
        try:
            feat = extract_feature(path, use_color=use_color, use_lbp=use_lbp)
            features_list.append(feat)
        except Exception as e:
            print(f"[WARNING] Lỗi khi trích feature từ {path}: {e}")
            # Tạo zero vector
            feat = np.zeros_like(features_list[0]) if features_list else np.zeros(100, dtype=np.float32)
            features_list.append(feat)

    features = np.array(features_list, dtype=np.float32)

    elapsed = time.time() - start_time
    print(f"[INFO] Feature extraction time: {elapsed:.2f}s")
    print(f"[INFO] Feature shape: {features.shape}")

    return features


def build_lsh_index(features):
    """
    Build LSH index từ features.

    Args:
        features: ndarray shape (N, D)

    Returns:
        index: LSHIndex
    """
    print("\n" + "="*60)
    print("STEP 3: BUILD LSH INDEX")
    print("="*60)

    N, D = features.shape
    print(f"[INFO] Num vectors: {N}, Dim: {D}")
    print(f"[INFO] LSH params: tables={cfg.num_tables}, planes={cfg.num_planes}")

    start_time = time.time()

    index = LSHIndex(
        num_tables=cfg.num_tables,
        num_planes=cfg.num_planes,
        dim=D,
        seed=cfg.lsh_seed
    )
    index.fit(features)

    elapsed = time.time() - start_time
    print(f"[INFO] Indexing time: {elapsed:.2f}s")

    return index


def main():
    args = parse_args()

    # Tạo artifacts folder
    mkdir_if_not_exists(cfg.artifacts_dir)

    # Step 1: Download dataset
    if args.download:
        download_dataset()

    # Step 2: Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    dataset_dir = Path(cfg.out_dir_dataset)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder '{dataset_dir}' không tồn tại!")
        print("[INFO] Hãy chạy với --download để tải dataset.")
        sys.exit(1)

    paths, labels, class_names = load_paths_labels(cfg.out_dir_dataset)
    print(f"[INFO] Loaded {len(paths)} images, {len(class_names)} classes")

    # Step 3: Extract features
    features_path = Path(cfg.features_path)
    meta_path = Path(cfg.meta_path)

    if args.skip_features and features_path.exists():
        print("\n[INFO] Skip feature extraction, loading existing features...")
        features = np.load(features_path)
        meta_df = pd.read_csv(meta_path)
        print(f"[INFO] Loaded features: {features.shape}")
    else:
        features = extract_all_features(paths, use_color=args.use_color, use_lbp=args.use_lbp)

        # Save features
        print(f"[INFO] Saving features to {features_path}...")
        np.save(features_path, features)

        # Save metadata
        print(f"[INFO] Saving metadata to {meta_path}...")
        meta_df = pd.DataFrame({
            'id': range(len(paths)),
            'path': paths,
            'label': labels,
            'class_name': [class_names[label] for label in labels]
        })
        meta_df.to_csv(meta_path, index=False)

    # Step 4: Build LSH index
    index_path = Path(cfg.index_path)

    if args.rebuild_index or not index_path.exists():
        index = build_lsh_index(features)

        # Save index
        print(f"[INFO] Saving LSH index to {index_path}...")
        index.save(index_path)
    else:
        print(f"\n[INFO] LSH index đã tồn tại tại {index_path}, skip build.")

    print("\n" + "="*60)
    print("OFFLINE BUILD HOÀN TẤT!")
    print("="*60)
    print(f"Features: {features_path}")
    print(f"Metadata: {meta_path}")
    print(f"LSH Index: {index_path}")
    print("="*60)


if __name__ == "__main__":
    main()
