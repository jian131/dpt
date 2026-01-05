"""
LSH Index with random hyperplanes (TỰ CODE)
"""
import numpy as np
import pickle

class LSHIndex:
    def __init__(self, num_tables, num_planes, dim, seed=42):
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.dim = dim
        self.seed = seed
        self.planes = self._make_planes()
        self.tables = [dict() for _ in range(num_tables)]
        self.num_vectors = 0

    def _make_planes(self):
        """Generate random hyperplanes"""
        np.random.seed(self.seed)
        planes = []
        for _ in range(self.num_tables):
            p = np.random.randn(self.num_planes, self.dim).astype(np.float32)
            p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
            planes.append(p)
        return planes

    def _hash(self, vec, planes):
        """Hash vector using random hyperplanes"""
        dots = np.dot(planes, vec)
        bits = (dots >= 0).astype(np.uint8)
        hash_val = 0
        for i, bit in enumerate(bits):
            if bit:
                hash_val |= (1 << i)
        return hash_val

    def fit(self, vectors):
        """Build index from vectors"""
        self.num_vectors = len(vectors)
        self.tables = [dict() for _ in range(self.num_tables)]

        for vid in range(len(vectors)):
            for tid in range(self.num_tables):
                h = self._hash(vectors[vid], self.planes[tid])
                if h not in self.tables[tid]:
                    self.tables[tid][h] = []
                self.tables[tid][h].append(vid)

    def query(self, vec):
        """Query candidates for vector"""
        candidates = set()
        for tid in range(self.num_tables):
            h = self._hash(vec, self.planes[tid])
            if h in self.tables[tid]:
                candidates.update(self.tables[tid][h])
        return candidates

    def save(self, path):
        """Save index to file"""
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
        """Load index from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        idx = LSHIndex(data['num_tables'], data['num_planes'], data['dim'], data['seed'])
        idx.planes = data['planes']
        idx.tables = data['tables']
        idx.num_vectors = data['num_vectors']
        return idx
