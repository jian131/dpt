"""
Evaluation: đánh giá Precision@K, Recall@K và so sánh Linear vs LSH.

Ground truth: ảnh cùng class là relevant (trừ chính nó).

Usage:
    python eval.py --k 10 --num_queries 50 --mode both
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config as cfg
from features import extract_feature
from similarity import pairwise_distance, topk_indices
from indexing import LSHIndex
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--k", type=int, default=cfg.eval_k, help="K for Precision@K")
    parser.add_argument("--num_queries", type=int, default=cfg.eval_num_queries, help="Number of query images")
    parser.add_argument("--mode", type=str, default="both", choices=["linear", "lsh", "both"], help="Evaluation mode")
    parser.add_argument("--metric", type=str, default=cfg.metric_default, choices=["l1", "l2", "chi2"], help="Distance metric")
    parser.add_argument("--use_color", action="store_true", default=True, help="Use color histogram")
    parser.add_argument("--use_lbp", action="store_true", default=True, help="Use LBP texture")
    parser.add_argument("--seed", type=int, default=cfg.eval_seed, help="Random seed")
    return parser.parse_args()


def load_artifacts():
    """
    Load features, metadata, và LSH index.
    """
    features_path = Path(cfg.features_path)
    meta_path = Path(cfg.meta_path)
    index_path = Path(cfg.index_path)

    if not features_path.exists():
        print(f"[ERROR] Features không tồn tại: {features_path}")
        sys.exit(1)

    features = np.load(features_path)
    meta_df = pd.read_csv(meta_path)

    index = None
    if index_path.exists():
        index = LSHIndex.load(index_path)

    return features, meta_df, index


def compute_precision_recall(topk_labels, gt_label, total_relevant):
    """
    Tính Precision@K và Recall@K.

    Args:
        topk_labels: list of labels, top-K results
        gt_label: int, ground truth label
        total_relevant: int, tổng số ảnh relevant trong database

    Returns:
        precision: float
        recall: float
    """
    K = len(topk_labels)

    # Đếm số correct trong top-K
    num_correct = sum(1 for label in topk_labels if label == gt_label)

    # Precision@K = correct / K
    precision = num_correct / K if K > 0 else 0.0

    # Recall@K = correct / total_relevant
    recall = num_correct / total_relevant if total_relevant > 0 else 0.0

    return precision, recall


def eval_linear(query_ids, features, labels, metric, k):
    """
    Evaluate với linear search.

    Returns:
        precisions: list of float
        recalls: list of float
        avg_time: float (ms)
    """
    precisions = []
    recalls = []
    times = []

    for query_id in tqdm(query_ids, desc="Linear eval"):
        query_vec = features[query_id]
        gt_label = labels[query_id]

        # Tính số relevant (trừ chính nó)
        total_relevant = sum(1 for label in labels if label == gt_label) - 1

        # Search
        start_time = time.time()
        distances = pairwise_distance(query_vec, features, metric=metric)

        # Loại bỏ chính query (set distance = inf)
        distances[query_id] = np.inf

        # Top-K
        topk_ids = topk_indices(distances, k)
        elapsed = (time.time() - start_time) * 1000
        times.append(elapsed)

        # Get labels
        topk_labels = [labels[i] for i in topk_ids]

        # Compute metrics
        prec, rec = compute_precision_recall(topk_labels, gt_label, total_relevant)
        precisions.append(prec)
        recalls.append(rec)

    avg_time = np.mean(times)
    return precisions, recalls, avg_time


def eval_lsh(query_ids, features, labels, index, metric, k):
    """
    Evaluate với LSH search.

    Returns:
        precisions: list of float
        recalls: list of float
        avg_time: float (ms)
    """
    precisions = []
    recalls = []
    times = []

    for query_id in tqdm(query_ids, desc="LSH eval"):
        query_vec = features[query_id]
        gt_label = labels[query_id]

        # Tính số relevant
        total_relevant = sum(1 for label in labels if label == gt_label) - 1

        # Search
        start_time = time.time()

        # Query LSH
        candidates = index.query(query_vec)

        # Nếu không có candidates, fallback linear
        if len(candidates) == 0:
            distances = pairwise_distance(query_vec, features, metric=metric)
            distances[query_id] = np.inf
            topk_ids = topk_indices(distances, k)
        else:
            # Remove query_id from candidates
            candidates.discard(query_id)
            candidate_ids = list(candidates)

            if len(candidate_ids) == 0:
                # Fallback linear
                distances = pairwise_distance(query_vec, features, metric=metric)
                distances[query_id] = np.inf
                topk_ids = topk_indices(distances, k)
            else:
                # Rerank candidates
                candidate_features = features[candidate_ids]
                distances = pairwise_distance(query_vec, candidate_features, metric=metric)
                local_topk_ids = topk_indices(distances, min(k, len(distances)))
                topk_ids = [candidate_ids[i] for i in local_topk_ids]

        elapsed = (time.time() - start_time) * 1000
        times.append(elapsed)

        # Get labels
        topk_labels = [labels[i] for i in topk_ids]

        # Compute metrics
        prec, rec = compute_precision_recall(topk_labels, gt_label, total_relevant)
        precisions.append(prec)
        recalls.append(rec)

    avg_time = np.mean(times)
    return precisions, recalls, avg_time


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load artifacts
    print("\n" + "="*60)
    print("LOADING ARTIFACTS")
    print("="*60)
    features, meta_df, index = load_artifacts()
    print(f"[INFO] Features: {features.shape}")
    print(f"[INFO] Metadata: {len(meta_df)} images")

    labels = meta_df['label'].tolist()

    # Select random queries
    num_samples = len(features)
    query_ids = np.random.choice(num_samples, size=min(args.num_queries, num_samples), replace=False)

    print(f"\n[INFO] Num queries: {len(query_ids)}")
    print(f"[INFO] K: {args.k}")
    print(f"[INFO] Metric: {args.metric}")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    results = {}

    if args.mode in ["linear", "both"]:
        print("\n--- LINEAR SEARCH ---")
        precisions, recalls, avg_time = eval_linear(query_ids, features, labels, args.metric, args.k)
        results['linear'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'time': avg_time
        }
        print(f"[RESULT] Linear - Precision@{args.k}: {results['linear']['precision']:.4f}")
        print(f"[RESULT] Linear - Recall@{args.k}: {results['linear']['recall']:.4f}")
        print(f"[RESULT] Linear - Avg query time: {results['linear']['time']:.2f} ms")

    if args.mode in ["lsh", "both"]:
        if index is None:
            print("[ERROR] LSH index không tồn tại!")
            sys.exit(1)

        print("\n--- LSH SEARCH ---")
        precisions, recalls, avg_time = eval_lsh(query_ids, features, labels, index, args.metric, args.k)
        results['lsh'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'time': avg_time
        }
        print(f"[RESULT] LSH - Precision@{args.k}: {results['lsh']['precision']:.4f}")
        print(f"[RESULT] LSH - Recall@{args.k}: {results['lsh']['recall']:.4f}")
        print(f"[RESULT] LSH - Avg query time: {results['lsh']['time']:.2f} ms")

    # Summary
    if args.mode == "both":
        print("\n" + "="*60)
        print("SO SÁNH LINEAR vs LSH")
        print("="*60)
        speedup = results['linear']['time'] / results['lsh']['time']
        precision_loss = results['linear']['precision'] - results['lsh']['precision']

        print(f"Speedup: {speedup:.2f}x")
        print(f"Precision loss: {precision_loss:.4f} ({precision_loss*100:.2f}%)")
        print("="*60)


if __name__ == "__main__":
    main()
