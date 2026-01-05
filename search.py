"""
Search: query-by-image, trả về Top-K ảnh giống nhất.

Mode:
    - linear: tìm kiếm tuyến tính trên toàn bộ database
    - lsh: dùng LSH index để tìm candidates rồi rerank

Usage:
    python search.py --query "dataset/accordion/0.jpg" --mode lsh --topk 10 --metric chi2
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import config as cfg
from features import extract_feature
from similarity import pairwise_distance, topk_indices
from indexing import LSHIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Search similar images")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--mode", type=str, default="lsh", choices=["linear", "lsh"], help="Search mode")
    parser.add_argument("--topk", type=int, default=cfg.topk_default, help="Number of results")
    parser.add_argument("--metric", type=str, default=cfg.metric_default, choices=["l1", "l2", "chi2"], help="Distance metric")
    parser.add_argument("--use_color", action="store_true", default=True, help="Use color histogram")
    parser.add_argument("--use_lbp", action="store_true", default=True, help="Use LBP texture")
    parser.add_argument("--no_show", action="store_true", help="Don't show result images")
    return parser.parse_args()


def load_artifacts():
    """
    Load features, metadata, và LSH index.

    Returns:
        features: ndarray (N, D)
        meta_df: pandas DataFrame
        index: LSHIndex hoặc None
    """
    print("\n" + "="*60)
    print("LOADING ARTIFACTS")
    print("="*60)

    features_path = Path(cfg.features_path)
    meta_path = Path(cfg.meta_path)
    index_path = Path(cfg.index_path)

    if not features_path.exists():
        print(f"[ERROR] Features không tồn tại: {features_path}")
        print("[INFO] Hãy chạy offline_build.py trước!")
        sys.exit(1)

    # Load features
    print(f"[INFO] Loading features from {features_path}...")
    features = np.load(features_path)
    print(f"[INFO] Features shape: {features.shape}")

    # Load metadata
    print(f"[INFO] Loading metadata from {meta_path}...")
    meta_df = pd.read_csv(meta_path)
    print(f"[INFO] Metadata: {len(meta_df)} images")

    # Load LSH index (optional)
    index = None
    if index_path.exists():
        print(f"[INFO] Loading LSH index from {index_path}...")
        index = LSHIndex.load(index_path)
        print(f"[INFO] LSH index loaded: {index.num_vectors} vectors")
    else:
        print("[WARNING] LSH index không tồn tại, chỉ dùng được mode=linear")

    return features, meta_df, index


def search_linear(query_vec, features, metric, topk):
    """
    Tìm kiếm tuyến tính trên toàn bộ database.

    Args:
        query_vec: ndarray (D,)
        features: ndarray (N, D)
        metric: str
        topk: int

    Returns:
        topk_ids: ndarray (topk,)
        topk_dists: ndarray (topk,)
        num_candidates: int
    """
    start_time = time.time()

    # Tính distance với tất cả vectors
    distances = pairwise_distance(query_vec, features, metric=metric)

    # Lấy Top-K
    topk_ids = topk_indices(distances, topk)
    topk_dists = distances[topk_ids]

    elapsed = (time.time() - start_time) * 1000  # ms

    return topk_ids, topk_dists, len(features), elapsed


def search_lsh(query_vec, features, index, metric, topk):
    """
    Tìm kiếm dùng LSH index.

    Args:
        query_vec: ndarray (D,)
        features: ndarray (N, D)
        index: LSHIndex
        metric: str
        topk: int

    Returns:
        topk_ids: ndarray (topk,)
        topk_dists: ndarray (topk,)
        num_candidates: int
    """
    start_time = time.time()

    # Query LSH index để lấy candidates
    candidates = index.query(query_vec)

    if len(candidates) == 0:
        print("[WARNING] LSH không trả về candidates, fallback sang linear search")
        return search_linear(query_vec, features, metric, topk)

    # Convert candidates sang list
    candidate_ids = list(candidates)

    # Lấy features của candidates
    candidate_features = features[candidate_ids]

    # Tính distance chỉ với candidates
    distances = pairwise_distance(query_vec, candidate_features, metric=metric)

    # Lấy Top-K trong candidates
    local_topk_ids = topk_indices(distances, min(topk, len(distances)))

    # Map về global ids
    topk_ids = np.array([candidate_ids[i] for i in local_topk_ids])
    topk_dists = distances[local_topk_ids]

    elapsed = (time.time() - start_time) * 1000  # ms

    return topk_ids, topk_dists, len(candidates), elapsed


def show_results(query_path, topk_paths, topk_dists, topk_labels):
    """
    Hiển thị kết quả tìm kiếm bằng matplotlib.

    Args:
        query_path: str
        topk_paths: list of str
        topk_dists: list of float
        topk_labels: list of str
    """
    K = len(topk_paths)

    # Tạo figure
    fig, axes = plt.subplots(2, (K + 1) // 2 + 1, figsize=(15, 6))
    axes = axes.flatten()

    # Show query image
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show top-K results
    for i in range(K):
        img = cv2.imread(topk_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"#{i+1}: {topk_labels[i]}\ndist={topk_dists[i]:.4f}", fontsize=10)
        axes[i + 1].axis('off')

    # Hide unused axes
    for i in range(K + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # Kiểm tra query image
    if not os.path.exists(args.query):
        print(f"[ERROR] Query image không tồn tại: {args.query}")
        sys.exit(1)

    # Load artifacts
    features, meta_df, index = load_artifacts()

    # Extract query feature
    print("\n" + "="*60)
    print("EXTRACT QUERY FEATURE")
    print("="*60)
    print(f"[INFO] Query image: {args.query}")

    start_time = time.time()
    query_vec = extract_feature(args.query, use_color=args.use_color, use_lbp=args.use_lbp)
    extract_time = (time.time() - start_time) * 1000
    print(f"[INFO] Feature extraction time: {extract_time:.2f} ms")
    print(f"[INFO] Feature dim: {len(query_vec)}")

    # Search
    print("\n" + "="*60)
    print(f"SEARCH MODE: {args.mode.upper()}")
    print("="*60)

    if args.mode == "linear":
        topk_ids, topk_dists, num_candidates, search_time = search_linear(
            query_vec, features, args.metric, args.topk
        )
    elif args.mode == "lsh":
        if index is None:
            print("[ERROR] LSH index không tồn tại!")
            sys.exit(1)
        topk_ids, topk_dists, num_candidates, search_time = search_lsh(
            query_vec, features, index, args.metric, args.topk
        )

    # Get results info
    topk_paths = meta_df.iloc[topk_ids]['path'].tolist()
    topk_labels = meta_df.iloc[topk_ids]['class_name'].tolist()

    # Print results
    print(f"[INFO] Search time: {search_time:.2f} ms")
    print(f"[INFO] Candidates: {num_candidates} / {len(features)}")
    print(f"[INFO] Metric: {args.metric}")
    print(f"\nTop-{args.topk} Results:")
    print("-" * 60)
    for i, (path, dist, label) in enumerate(zip(topk_paths, topk_dists, topk_labels)):
        print(f"#{i+1}: {label:20s} | dist={dist:.6f} | {path}")
    print("-" * 60)

    # Show results
    if not args.no_show:
        show_results(args.query, topk_paths, topk_dists, topk_labels)


if __name__ == "__main__":
    main()
