"""
LSH Index (Locality Sensitive Hashing) - TỰ CODE
Dùng random hyperplanes để hash vectors vào buckets.

Mục tiêu: giảm thời gian tìm kiếm từ O(N) xuống O(N/num_tables) trung bình.
"""
import numpy as np
import pickle


class LSHIndex:
    """
    LSH Index dùng random hyperplanes.

    - num_tables: số bảng hash (càng nhiều càng chính xác nhưng chậm hơn)
    - num_planes: số hyperplanes mỗi bảng (càng nhiều càng ít collision)
    - dim: số chiều của feature vector
    """

    def __init__(self, num_tables, num_planes, dim, seed=42):
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.dim = dim
        self.seed = seed

        # Tạo random planes
        self.planes = self._make_planes()

        # Hash tables: list of dict[hash_value -> list[ids]]
        self.tables = [dict() for _ in range(num_tables)]

        # Metadata
        self.num_vectors = 0

    def _make_planes(self):
        """
        Tạo random hyperplanes cho mỗi table.

        Returns:
            planes: list of ndarray, mỗi shape (num_planes, dim)
        """
        np.random.seed(self.seed)
        planes = []
        for _ in range(self.num_tables):
            # Random normal distribution
            plane = np.random.randn(self.num_planes, self.dim).astype(np.float32)
            # Normalize mỗi plane
            plane = plane / (np.linalg.norm(plane, axis=1, keepdims=True) + 1e-12)
            planes.append(plane)
        return planes

    def _hash(self, vec, planes):
        """
        Hash vector bằng random hyperplanes.

        Args:
            vec: ndarray shape (D,)
            planes: ndarray shape (num_planes, D)

        Returns:
            hash_value: int (bit packing)
        """
        # Tính dot product với từng plane
        dots = np.dot(planes, vec)  # shape (num_planes,)

        # Bits: 1 nếu dot >= 0, else 0
        bits = (dots >= 0).astype(np.uint8)

        # Chuyển bits thành int (bit packing)
        hash_value = 0
        for i, bit in enumerate(bits):
            if bit:
                hash_value |= (1 << i)

        return hash_value

    def fit(self, vectors):
        """
        Build index từ list vectors.

        Args:
            vectors: ndarray shape (N, D)
        """
        N = len(vectors)
        self.num_vectors = N

        # Clear tables
        self.tables = [dict() for _ in range(self.num_tables)]

        # Hash từng vector vào từng table
        for vec_id in range(N):
            vec = vectors[vec_id]

            for table_idx in range(self.num_tables):
                hash_val = self._hash(vec, self.planes[table_idx])

                # Thêm vec_id vào bucket
                if hash_val not in self.tables[table_idx]:
                    self.tables[table_idx][hash_val] = []
                self.tables[table_idx][hash_val].append(vec_id)

    def query(self, vec):
        """
        Query vector và trả về candidate ids.

        Args:
            vec: ndarray shape (D,)

        Returns:
            candidates: set of int (vec ids)
        """
        candidates = set()

        for table_idx in range(self.num_tables):
            hash_val = self._hash(vec, self.planes[table_idx])

            # Lấy bucket
            if hash_val in self.tables[table_idx]:
                bucket = self.tables[table_idx][hash_val]
                candidates.update(bucket)

        return candidates

    def save(self, path):
        """
        Lưu index ra file.

        Args:
            path: đường dẫn file .pkl
        """
        data = {
            'num_tables': self.num_tables,
            'num_planes': self.num_planes,
            'dim': self.dim,
            'seed': self.seed,
            'planes': self.planes,
            'tables': self.tables,
            'num_vectors': self.num_vectors
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        """
        Load index từ file.

        Args:
            path: đường dẫn file .pkl

        Returns:
            index: LSHIndex object
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = LSHIndex(
            num_tables=data['num_tables'],
            num_planes=data['num_planes'],
            dim=data['dim'],
            seed=data['seed']
        )
        index.planes = data['planes']
        index.tables = data['tables']
        index.num_vectors = data['num_vectors']

        return index
