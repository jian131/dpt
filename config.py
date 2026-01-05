"""
Cấu hình mặc định cho CBIR project
"""
import os

# ============ DATASET ============
dataset_name = "caltech101"
out_dir_dataset = "dataset"
export_split = "train"
limit_per_class = 80
max_classes = 30
image_size = (256, 256)

# ============ FEATURE EXTRACTION ============
# HSV histogram: H=16, S=4, V=4 => K = 16*4*4 = 256 bins per cell
bins_H = 16
bins_S = 4
bins_V = 4
K_hsv = bins_H * bins_S * bins_V  # 256

# Grid for spatial color histogram
grid = (3, 3)  # 3x3 grid => 9 cells
color_dim = grid[0] * grid[1] * K_hsv  # 9 * 256 = 2304

# LBP: basic 3x3 => 256 bins
lbp_dim = 256

# Feature weights
w_color = 0.6
w_lbp = 0.4

# ============ SIMILARITY ============
metric_default = "chi2"  # chi2, l1, l2
topk_default = 10

# ============ LSH INDEXING ============
num_tables = 8
num_planes = 12
lsh_seed = 42

# ============ ARTIFACTS ============
artifacts_dir = "artifacts"
features_path = os.path.join(artifacts_dir, "features.npy")
meta_path = os.path.join(artifacts_dir, "meta.csv")
index_path = os.path.join(artifacts_dir, "lsh_index.pkl")

# ============ EVALUATION ============
eval_k = 10
eval_num_queries = 50
eval_seed = 42
