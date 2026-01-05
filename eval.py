"""
Evaluation: Precision@K and Recall@K
"""
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config as cfg
from lsh import LSHIndex

# Import distance functions from search.py
import sys
sys.path.insert(0, '.')
from search import pairwise_distance, topk_indices

def evaluate(query_ids, features, labels, index, metric, k, use_lsh):
    """Evaluate retrieval performance"""
    precisions, recalls, times = [], [], []
    
    for qid in tqdm(query_ids, desc=f"{'LSH' if use_lsh else 'Linear'} eval"):
        query_vec = features[qid]
        gt_label = labels[qid]
        total_relevant = sum(1 for l in labels if l == gt_label) - 1
        
        start = time.time()
        
        if use_lsh:
            candidates = index.query(query_vec)
            candidates.discard(qid)
            
            if not candidates:
                distances = pairwise_distance(query_vec, features, metric)
                distances[qid] = np.inf
                topk_ids = topk_indices(distances, k)
            else:
                cand_ids = list(candidates)
                cand_feats = features[cand_ids]
                distances = pairwise_distance(query_vec, cand_feats, metric)
                local_topk = topk_indices(distances, min(k, len(distances)))
                topk_ids = [cand_ids[i] for i in local_topk]
        else:
            distances = pairwise_distance(query_vec, features, metric)
            distances[qid] = np.inf
            topk_ids = topk_indices(distances, k)
        
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        topk_labels = [labels[i] for i in topk_ids]
        correct = sum(1 for l in topk_labels if l == gt_label)
        
        precisions.append(correct / k if k > 0 else 0)
        recalls.append(correct / total_relevant if total_relevant > 0 else 0)
    
    return np.mean(precisions), np.mean(recalls), np.mean(times)

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--k", type=int, default=cfg.eval_k)
    parser.add_argument("--num_queries", type=int, default=cfg.eval_num_queries)
    parser.add_argument("--mode", type=str, default="both", choices=["linear", "lsh", "both"])
    parser.add_argument("--metric", type=str, default=cfg.metric_default, choices=["l1", "l2", "chi2"])
    parser.add_argument("--seed", type=int, default=cfg.eval_seed)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load
    print(f"\n{'='*60}\nLOAD ARTIFACTS\n{'='*60}")
    features = np.load(cfg.features_path)
    meta_df = pd.read_csv(cfg.meta_path)
    labels = meta_df['label'].tolist()
    index = LSHIndex.load(cfg.index_path) if Path(cfg.index_path).exists() else None
    
    query_ids = np.random.choice(len(features), min(args.num_queries, len(features)), replace=False)
    print(f"Features: {features.shape} | Queries: {len(query_ids)} | K: {args.k}")
    
    # Evaluate
    print(f"\n{'='*60}\nEVALUATION\n{'='*60}")
    results = {}
    
    if args.mode in ["linear", "both"]:
        print("\n--- LINEAR ---")
        prec, rec, t = evaluate(query_ids, features, labels, None, args.metric, args.k, False)
        results['linear'] = {'precision': prec, 'recall': rec, 'time': t}
        print(f"Precision@{args.k}: {prec:.4f}")
        print(f"Recall@{args.k}: {rec:.4f}")
        print(f"Avg time: {t:.2f}ms")
    
    if args.mode in ["lsh", "both"]:
        if index is None:
            print("\nLSH index not found!")
            return
        
        print("\n--- LSH ---")
        prec, rec, t = evaluate(query_ids, features, labels, index, args.metric, args.k, True)
        results['lsh'] = {'precision': prec, 'recall': rec, 'time': t}
        print(f"Precision@{args.k}: {prec:.4f}")
        print(f"Recall@{args.k}: {rec:.4f}")
        print(f"Avg time: {t:.2f}ms")
    
    # Compare
    if args.mode == "both":
        print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
        speedup = results['linear']['time'] / results['lsh']['time']
        prec_loss = results['linear']['precision'] - results['lsh']['precision']
        print(f"Speedup: {speedup:.2f}x")
        print(f"Precision loss: {prec_loss:.4f} ({prec_loss*100:.2f}%)")
        print("="*60)

if __name__ == "__main__":
    main()
