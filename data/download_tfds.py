"""
Download TensorFlow Datasets và export ảnh ra folder theo class.
BẮT BUỘC: Dùng TFDS để tải dataset và lưu thành dataset/<class_name>/<id>.jpg

Usage:
    python data/download_tfds.py --dataset caltech101 --out dataset --split train --limit_per_class 80 --max_classes 30
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds

# Import config mặc định
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Download TFDS dataset and export images")
    parser.add_argument("--dataset", type=str, default=cfg.dataset_name, help="TFDS dataset name")
    parser.add_argument("--out", type=str, default=cfg.out_dir_dataset, help="Output directory")
    parser.add_argument("--split", type=str, default=cfg.export_split, help="Dataset split (train/test)")
    parser.add_argument("--limit_per_class", type=int, default=cfg.limit_per_class, help="Max images per class")
    parser.add_argument("--max_classes", type=int, default=cfg.max_classes, help="Max number of classes")
    parser.add_argument("--image_size", type=int, default=cfg.image_size[0], help="Image size (square)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)

    # Kiểm tra nếu đã tồn tại và không force
    if out_dir.exists() and not args.force:
        print(f"[INFO] Dataset folder '{out_dir}' đã tồn tại. Dùng --force để ghi đè.")
        return

    # Tạo thư mục output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Đang tải dataset '{args.dataset}' split '{args.split}'...")

    # Load TensorFlow Datasets
    try:
        ds, info = tfds.load(
            args.dataset,
            split=args.split,
            with_info=True,
            shuffle_files=False
        )
    except Exception as e:
        print(f"[ERROR] Không thể tải dataset: {e}")
        sys.exit(1)

    # Lấy danh sách tên class
    if 'label' in info.features:
        label_names = info.features['label'].names
    else:
        print("[ERROR] Dataset không có field 'label'")
        sys.exit(1)

    print(f"[INFO] Dataset có {len(label_names)} classes")

    # Giới hạn số class
    num_classes = min(args.max_classes, len(label_names))
    label_names = label_names[:num_classes]
    print(f"[INFO] Chỉ export {num_classes} classes đầu tiên")

    # Counter cho mỗi class
    class_counters = {i: 0 for i in range(num_classes)}
    total_exported = 0

    # Duyệt qua dataset
    print(f"[INFO] Bắt đầu export ảnh (max {args.limit_per_class} ảnh/class)...")

    for example in tqdm(ds, desc="Exporting"):
        label = int(example['label'].numpy())

        # Chỉ lấy các class trong phạm vi
        if label >= num_classes:
            continue

        # Kiểm tra đã đủ limit chưa
        if class_counters[label] >= args.limit_per_class:
            continue

        # Lấy ảnh
        image = example['image'].numpy()

        # Resize ảnh
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize((args.image_size, args.image_size), Image.LANCZOS)

        # Đường dẫn lưu
        class_name = label_names[label]
        class_dir = out_dir / class_name
        class_dir.mkdir(exist_ok=True)

        img_id = class_counters[label]
        img_path = class_dir / f"{img_id}.jpg"

        # Lưu ảnh
        img_pil.convert('RGB').save(img_path, 'JPEG', quality=95)

        class_counters[label] += 1
        total_exported += 1

        # Kiểm tra đã đủ tất cả class chưa
        if all(count >= args.limit_per_class for count in class_counters.values()):
            break

    # Thống kê
    print("\n" + "="*60)
    print("THỐNG KÊ EXPORT")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Output folder: {out_dir.absolute()}")
    print(f"Classes exported: {num_classes}")
    print(f"Total images: {total_exported}")
    print(f"Images per class:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {class_counters[i]}")
    print("="*60)


if __name__ == "__main__":
    main()
