"""
GUI đơn giản cho CBIR system sử dụng Tkinter
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
import csv
from pathlib import Path

from config import *
from features import extract_feature
from search import search_lsh, search_linear
from lsh import LSHIndex
import config as cfg


class CBIRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR - Content-Based Image Retrieval")
        self.root.geometry("1200x700")

        # Load artifacts
        self.load_artifacts()

        # UI Components
        self.create_widgets()

        self.query_image_path = None

    def load_artifacts(self):
        """Load features, metadata, và LSH index"""
        try:
            self.features = np.load(cfg.features_path)

            # Load CSV manually (avoid pandas hang)
            self.meta = []
            with open(cfg.meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.meta.append(row)

            self.lsh_index = LSHIndex.load(cfg.index_path)
            messagebox.showinfo("Success", f"Loaded {len(self.features)} images")
        except Exception as e:
            messagebox.showerror("Error", f"Không load được artifacts:\n{e}")
            self.root.quit()

    def create_widgets(self):
        """Tạo UI components"""
        # Top frame - Query controls
        top_frame = tk.Frame(self.root, bg="#f0f0f0", height=100)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Query image display
        self.query_label = tk.Label(top_frame, text="Query Image", bg="#f0f0f0",
                                     font=("Arial", 10, "bold"))
        self.query_label.grid(row=0, column=0, padx=5, pady=5)

        self.query_canvas = tk.Canvas(top_frame, width=150, height=150, bg="white",
                                       relief=tk.SUNKEN, bd=2)
        self.query_canvas.grid(row=1, column=0, padx=5, pady=5, rowspan=3)

        # Controls
        tk.Button(top_frame, text="Chọn Ảnh Query", command=self.select_query_image,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  width=15, height=2).grid(row=1, column=1, padx=10, pady=5)

        # Search mode
        tk.Label(top_frame, text="Search Mode:", bg="#f0f0f0",
                 font=("Arial", 9)).grid(row=2, column=1, sticky=tk.W, padx=10)

        self.search_mode = tk.StringVar(value="lsh")
        tk.Radiobutton(top_frame, text="LSH (Nhanh)", variable=self.search_mode,
                       value="lsh", bg="#f0f0f0").grid(row=2, column=2, sticky=tk.W)
        tk.Radiobutton(top_frame, text="Linear (Chính xác)", variable=self.search_mode,
                       value="linear", bg="#f0f0f0").grid(row=3, column=2, sticky=tk.W)

        # TopK
        tk.Label(top_frame, text="Top-K:", bg="#f0f0f0",
                 font=("Arial", 9)).grid(row=2, column=3, sticky=tk.W, padx=10)
        self.topk_var = tk.IntVar(value=10)
        tk.Spinbox(top_frame, from_=5, to=50, textvariable=self.topk_var,
                   width=5).grid(row=2, column=4, sticky=tk.W)

        # Search button
        tk.Button(top_frame, text="🔍 Search", command=self.search,
                  bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
                  width=12, height=2).grid(row=1, column=5, padx=10, pady=5, rowspan=2)

        # Results info
        self.info_label = tk.Label(top_frame, text="", bg="#f0f0f0",
                                    font=("Arial", 9), fg="blue")
        self.info_label.grid(row=3, column=1, columnspan=4, sticky=tk.W, padx=10)

        # Results frame
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas for results
        self.results_canvas = tk.Canvas(results_frame, yscrollcommand=scrollbar.set)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_canvas.yview)

        # Frame inside canvas
        self.results_container = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_container, anchor=tk.NW)

        # Bind resize
        self.results_container.bind("<Configure>",
                                    lambda e: self.results_canvas.configure(
                                        scrollregion=self.results_canvas.bbox("all")))

    def select_query_image(self):
        """Chọn ảnh query"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh query",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )

        if file_path:
            self.query_image_path = file_path
            self.display_query_image(file_path)

    def display_query_image(self, image_path):
        """Hiển thị ảnh query"""
        try:
            img = Image.open(image_path)
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)

            self.query_canvas.delete("all")
            self.query_canvas.create_image(75, 75, image=photo)
            self.query_canvas.image = photo  # Keep reference

            # Show filename
            filename = Path(image_path).name
            self.query_label.config(text=f"Query: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Không load được ảnh:\n{e}")

    def search(self):
        """Thực hiện search"""
        if not self.query_image_path:
            messagebox.showwarning("Warning", "Vui lòng chọn ảnh query!")
            return

        try:
            # Extract feature
            query_feat = extract_feature(self.query_image_path, cfg)

            # Search
            mode = self.search_mode.get()
            topk = self.topk_var.get()

            if mode == "lsh":
                indices, distances, num_cand, search_time = search_lsh(
                    query_feat, self.features, self.lsh_index, 'chi2', topk
                )
            else:
                indices, distances, num_cand, search_time = search_linear(
                    query_feat, self.features, 'chi2', topk
                )

            # Display results
            self.display_results(indices, distances, search_time, mode)

        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi search:\n{e}")

    def display_results(self, indices, distances, search_time, mode):
        """Hiển thị kết quả search"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()

        # Info
        self.info_label.config(
            text=f"✓ Found {len(indices)} results in {search_time:.2f}ms ({mode.upper()} mode)"
        )

        # Display results in grid
        cols = 5
        for idx, (i, dist) in enumerate(zip(indices, distances)):
            row = idx // cols
            col = idx % cols

            # Frame for each result
            result_frame = tk.Frame(self.results_container, relief=tk.RAISED, bd=2)
            result_frame.grid(row=row, column=col, padx=5, pady=5)

            # Image
            img_path = self.meta[i]['path']
            try:
                img = Image.open(img_path)
                img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img)

                img_label = tk.Label(result_frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack()
            except:
                tk.Label(result_frame, text="[Image Error]", width=20, height=10).pack()

            # Info
            class_name = self.meta[i]['class_name']
            tk.Label(result_frame, text=f"#{idx+1}: {class_name}",
                     font=("Arial", 9, "bold")).pack()
            tk.Label(result_frame, text=f"Distance: {dist:.3f}",
                     font=("Arial", 8), fg="gray").pack()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CBIRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
