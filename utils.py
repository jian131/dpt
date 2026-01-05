"""
Utility functions: timer, mkdir, seed, etc.
"""
import os
import time
import random
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(name="Operation"):
    """
    Context manager để đo thời gian.

    Usage:
        with timer("Extract features"):
            # code
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = (end - start) * 1000  # ms
    print(f"[TIMER] {name}: {elapsed:.2f} ms")


def mkdir_if_not_exists(path):
    """
    Tạo folder nếu chưa tồn tại.

    Args:
        path: đường dẫn folder
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    """
    Set random seed cho reproducibility.

    Args:
        seed: int
    """
    random.seed(seed)
    np.random.seed(seed)


def format_time(seconds):
    """
    Format thời gian (seconds) thành string dễ đọc.

    Args:
        seconds: float

    Returns:
        str: ví dụ "1.23s" hoặc "123.45ms"
    """
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds * 1000:.2f}ms"
