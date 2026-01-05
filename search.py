"""
Search similar images (query-by-image)
"""
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

import config as cfg
from features import extract_feature
from lsh import LSHIndex

# ============ DISTANCE METRICS (TỰ CODE) ============

def chi2_distance(a, b, eps=1e-10):
    """Chi-square distance for histograms"""
    diff = a - b
    return 0.5 * np.sum((diff * diff) / (a + b + eps), axis=-1)

def l1_distance(a, b):
    """Manhattan distance"""
    return np.sum(np.abs(a - b), axis=-1)

def l2_distance(a, b):
    """Euclidean distance"""
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

def pairwise_distance(query, database, metric="chi2"):
    """Compute distances between query and all database vectors"""
    if metric == "chi2":
        return chi2_distance(query, database)
    elif metric == "l1":
        return l1_distance(query, database)
    elif metric == "l2":
        return l2_distance(query, database)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def topk_indices(distances, k):
    """Get top-K smallest distance indices"""
    k = min(k, len(distances))
    idx = np.argpartition(distances, k-1)[:k]
    return idx[np.argsort(distances[idx])]

# ============ SEARCH ============

def search_linear(query_vec, features, metric, k):
    """Linear search on full database"""
    start = time.time()
    distances = pairwise_distance(query_vec, features, metric)
    topk_ids = topk_indices(distances, k)
    elapsed = (time.time() - start) * 1000
    return topk_ids, distances[topk_ids], len(features), elapsed

def search_lsh(query_vec, features, index, metric, k):
    """LSH search with reranking"""
    start = time.time()
    candidates = index.query(query_vec)

    if not candidates:
        return search_linear(query_vec, features, metric, k)

    cand_ids = list(candidates)
    cand_feats = features[cand_ids]
    distances = pairwise_distance(query_vec, cand_feats, metric)

    local_topk = topk_indices(distances, min(k, len(distances)))
    topk_ids = np.array([cand_ids[i] for i in local_topk])
    topk_dists = distances[local_topk]

    elapsed = (time.time() - start) * 1000
    return topk_ids, topk_dists, len(candidates), elapsed

# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Search similar images")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", type=str, default="lsh", choices=["linear", "lsh"])
    parser.add_argument("--topk", type=int, default=cfg.topk_default)
    parser.add_argument("--metric", type=str, default=cfg.metric_default, choices=["l1", "l2", "chi2"])
    args = parser.parse_args()

    # Load artifacts
    print(f"\n{'='*60}\nLOAD ARTIFACTS\n{'='*60}")
    features = np.load(cfg.features_path)
    meta_df = pd.read_csv(cfg.meta_path)
    index = LSHIndex.load(cfg.index_path) if Path(cfg.index_path).exists() else None
    print(f"Features: {features.shape} | Images: {len(meta_df)}")

    # Extract query feature
    print(f"\n{'='*60}\nQUERY: {args.query}\n{'='*60}")
    start = time.time()
    query_vec = extract_feature(args.query, cfg)
    print(f"Feature extraction: {(time.time()-start)*1000:.1f}ms | Dim: {len(query_vec)}")

    # Search
    print(f"\n{'='*60}\nSEARCH ({args.mode.upper()})\n{'='*60}")
    if args.mode == "linear":
        topk_ids, topk_dists, num_cand, search_time = search_linear(
            query_vec, features, args.metric, args.topk
        )
    else:
        if index is None:
            print("LSH index not found! Using linear search.")
            args.mode = "linear"
            topk_ids, topk_dists, num_cand, search_time = search_linear(
                query_vec, features, args.metric, args.topk
            )
        else:
            topk_ids, topk_dists, num_cand, search_time = search_lsh(
                query_vec, features, index, args.metric, args.topk
            )

    # Results
    topk_paths = meta_df.iloc[topk_ids]['path'].tolist()
    topk_labels = meta_df.iloc[topk_ids]['class_name'].tolist()

    print(f"Search time: {search_time:.1f}ms | Candidates: {num_cand}/{len(features)}")
    print(f"\nTop-{args.topk} Results:")
    print("-" * 60)
    for i, (path, dist, label) in enumerate(zip(topk_paths, topk_dists, topk_labels)):
        print(f"#{i+1}: {label:15s} | dist={dist:.6f} | {path}")
    print("-" * 60)

if __name__ == "__main__":
    main()
