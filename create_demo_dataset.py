"""
Tạo demo dataset đơn giản để test system (không cần TFDS).
Tạo ảnh synthetic với các màu khác nhau.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_demo_dataset(output_dir="dataset", num_classes=10, images_per_class=30, image_size=256):
    """
    Tạo demo dataset với ảnh synthetic.
    Mỗi class sẽ có màu chủ đạo khác nhau.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Định nghĩa màu cho mỗi class (RGB)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 128),    # Teal
        (128, 128, 128),  # Gray
    ]

    class_names = [
        "red", "green", "blue", "yellow", "magenta",
        "cyan", "orange", "purple", "teal", "gray"
    ]

    print(f"[INFO] Tạo demo dataset: {num_classes} classes, {images_per_class} ảnh/class")
    print(f"[INFO] Output: {output_path.absolute()}")

    total = 0

    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        class_color = colors[class_idx]
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

        for img_idx in range(images_per_class):
            # Tạo ảnh với màu chủ đạo + thêm noise và shapes
            img = Image.new('RGB', (image_size, image_size), color=class_color)
            draw = ImageDraw.Draw(img)

            # Thêm một số variations
            # Thêm hình tròn hoặc vuông ngẫu nhiên
            np.random.seed(class_idx * 1000 + img_idx)

            for _ in range(3):  # Vẽ 3 shapes ngẫu nhiên
                x = np.random.randint(0, image_size - 50)
                y = np.random.randint(0, image_size - 50)
                w = np.random.randint(20, 60)
                h = np.random.randint(20, 60)

                # Màu shape: biến thể của màu chủ
                r = min(255, max(0, class_color[0] + np.random.randint(-50, 50)))
                g = min(255, max(0, class_color[1] + np.random.randint(-50, 50)))
                b = min(255, max(0, class_color[2] + np.random.randint(-50, 50)))
                shape_color = (r, g, b)

                if np.random.rand() > 0.5:
                    draw.ellipse([x, y, x+w, y+h], fill=shape_color)
                else:
                    draw.rectangle([x, y, x+w, y+h], fill=shape_color)

            # Thêm texture noise
            img_array = np.array(img)
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Lưu ảnh
            img_path = class_dir / f"{img_idx}.jpg"
            img.save(img_path, 'JPEG', quality=95)
            total += 1

        print(f"  Created {images_per_class} images for class '{class_name}'")

    print(f"\n[SUCCESS] Đã tạo {total} ảnh trong {num_classes} classes")
    print("="*60)

if __name__ == "__main__":
    create_demo_dataset(
        output_dir="dataset",
        num_classes=10,
        images_per_class=30,
        image_size=256
    )
