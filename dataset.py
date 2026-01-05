"""
Dataset utilities: load paths và labels từ folder structure.

Folder structure:
    dataset/
        class1/
            0.jpg
            1.jpg
            ...
        class2/
            0.jpg
            ...
"""
import os
from pathlib import Path


def load_paths_labels(root_dir):
    """
    Load image paths và labels từ dataset folder.

    Args:
        root_dir: đường dẫn đến folder dataset (có các subfolder class)

    Returns:
        paths: list of str, đường dẫn ảnh
        labels: list of int, label index
        class_names: list of str, tên class (sorted)
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"Dataset folder không tồn tại: {root_dir}")

    # Lấy danh sách class (sorted)
    class_names = sorted([d.name for d in root_path.iterdir() if d.is_dir()])

    if len(class_names) == 0:
        raise ValueError(f"Không tìm thấy class nào trong {root_dir}")

    # Map class_name -> label index
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    # Thu thập paths và labels
    paths = []
    labels = []

    for class_name in class_names:
        class_dir = root_path / class_name
        label = class_to_label[class_name]

        # Lấy tất cả file jpg
        image_files = sorted(class_dir.glob("*.jpg"))

        for img_file in image_files:
            paths.append(str(img_file))
            labels.append(label)

    return paths, labels, class_names
