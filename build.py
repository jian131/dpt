"""
Build offline: extract features and build LSH index
"""
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config as cfg
from features import extract_feature
from lsh import LSHIndex

# ============ DATASET LOADING ============

def load_dataset(root_dir):
    """Load image paths and labels from folder structure"""
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"Dataset not found: {root_dir}")

    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No classes found in {root_dir}")

    paths, labels = [], []
    for label, cname in enumerate(class_names):
        for img in sorted((root / cname).glob("*.jpg")):
            paths.append(str(img))
            labels.append(label)

    return paths, labels, class_names

# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Build features and index")
    parser.add_argument("--dataset", type=str, default=cfg.out_dir_dataset)
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--rebuild_index", action="store_true")
    args = parser.parse_args()

    Path(cfg.artifacts_dir).mkdir(exist_ok=True)

    # Load dataset
    print(f"\n{'='*60}\nLOAD DATASET\n{'='*60}")
    paths, labels, class_names = load_dataset(args.dataset)
    print(f"Found {len(paths)} images, {len(class_names)} classes")

    # Extract features
    features_path = Path(cfg.features_path)
    if args.skip_features and features_path.exists():
        print(f"\nLoading features from {features_path}...")
        features = np.load(features_path)
    else:
        print(f"\n{'='*60}\nEXTRACT FEATURES\n{'='*60}")
        start = time.time()
        features = []
        for p in tqdm(paths, desc="Extracting"):
            try:
                features.append(extract_feature(p, cfg))
            except Exception as e:
                print(f"Error {p}: {e}")
                features.append(np.zeros(2560, dtype=np.float32))

        features = np.array(features, dtype=np.float32)
        print(f"Time: {time.time()-start:.1f}s | Shape: {features.shape}")

        np.save(features_path, features)
        pd.DataFrame({
            'id': range(len(paths)),
            'path': paths,
            'label': labels,
            'class_name': [class_names[l] for l in labels]
        }).to_csv(cfg.meta_path, index=False)
        print(f"Saved: {features_path}, {cfg.meta_path}")

    # Build LSH index
    index_path = Path(cfg.index_path)
    if args.rebuild_index or not index_path.exists():
        print(f"\n{'='*60}\nBUILD LSH INDEX\n{'='*60}")
        start = time.time()
        index = LSHIndex(cfg.num_tables, cfg.num_planes, features.shape[1], cfg.lsh_seed)
        index.fit(features)
        index.save(index_path)
        print(f"Time: {time.time()-start:.1f}s | Saved: {index_path}")

    print(f"\n{'='*60}\nDONE!\n{'='*60}")

if __name__ == "__main__":
    main()
