"""
Distance metrics (TỰ CODE)
- L1 (Manhattan)
- L2 (Euclidean)
- Chi-square

Input: feature vectors
Output: distance (scalar hoặc array)
"""
import numpy as np


def l1_distance(a, b):
    """
    L1 distance (Manhattan distance).

    Args:
        a, b: ndarray, feature vectors

    Returns:
        distance: float hoặc ndarray nếu b là matrix
    """
    return np.sum(np.abs(a - b), axis=-1)


def l2_distance(a, b):
    """
    L2 distance (Euclidean distance).

    Args:
        a, b: ndarray, feature vectors

    Returns:
        distance: float hoặc ndarray nếu b là matrix
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=-1))


def chi2_distance(a, b, eps=1e-10):
    """
    Chi-square distance: 0.5 * sum((a - b)^2 / (a + b + eps))

    Args:
        a, b: ndarray, feature vectors (đã normalize, non-negative)
        eps: epsilon để tránh chia 0

    Returns:
        distance: float hoặc ndarray nếu b là matrix
    """
    diff = a - b
    sum_ab = a + b + eps
    chi2 = 0.5 * np.sum((diff * diff) / sum_ab, axis=-1)
    return chi2


def pairwise_distance(query, database, metric="l2"):
    """
    Tính distance giữa query và tất cả vectors trong database.

    Args:
        query: ndarray shape (D,)
        database: ndarray shape (N, D)
        metric: "l1", "l2", hoặc "chi2"

    Returns:
        distances: ndarray shape (N,)
    """
    if metric == "l1":
        return l1_distance(query, database)
    elif metric == "l2":
        return l2_distance(query, database)
    elif metric == "chi2":
        return chi2_distance(query, database)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def topk_indices(distances, k):
    """
    Trả về indices của k distances nhỏ nhất (sorted).

    Args:
        distances: ndarray shape (N,)
        k: số lượng top results

    Returns:
        indices: ndarray shape (k,), sorted by distance
    """
    k = min(k, len(distances))
    # np.argpartition: O(n) nhưng không sort hoàn toàn
    # Sau đó sort lại k phần tử
    idx_partition = np.argpartition(distances, k-1)[:k]
    idx_sorted = idx_partition[np.argsort(distances[idx_partition])]
    return idx_sorted
