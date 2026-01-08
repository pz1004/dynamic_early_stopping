import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import time

class BaseSorter:
    def __init__(self, rng=None):
        """
        Base sorter that returns random permutation.

        Args:
            rng: numpy random generator. If None, uses np.random.default_rng().
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X):
        pass

    def get_sorted_indices(self, q, X):
        """Returns indices of X sorted by likely proximity to q."""
        # Default: Random permutation (Baseline)
        n = X.shape[0]
        return self.rng.permutation(n)


class RandomOrderSorter(BaseSorter):
    """Random permutation sorter to satisfy Beta-Geometric assumptions."""

    def get_sorted_indices(self, q, X):
        n = X.shape[0]
        return self.rng.permutation(n)


class PCASorter(BaseSorter):
    def __init__(self, n_components=16):
        """
        Sorts based on distance in a reduced subspace.
        
        Args:
            n_components: Number of dimensions to keep (e.g., 16 vs 128).
                          Lower = Faster sort, Less accurate.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.X_reduced = None

    def fit(self, X):
        print(f"Training PCA Sorter ({X.shape[1]} -> {self.n_components} dims)...")
        t0 = time.time()
        self.X_reduced = self.pca.fit_transform(X)
        print(f"PCA fit done in {time.time() - t0:.2f}s")
        return self

    def get_sorted_indices(self, q, X=None):
        """
        1. Project query q to subspace.
        2. Compute approximate distances (vectorized).
        3. Return argsort.
        """
        # Project query (1, D) -> (1, d)
        q_reduced = self.pca.transform(q.reshape(1, -1))
        
        # Compute subspace distances (Fast vectorized op on small dims)
        # ||x - q||^2
        diff = self.X_reduced - q_reduced
        # We only need the order, so squared euclidean is fine
        dists_sq = np.sum(diff**2, axis=1)
        
        # Sort indices (argsort is O(N log N), but N is small enough for this to be fast)
        # Optimization: use np.argpartition if we only need top P%? 
        # For DES, we want a full valid sequence, so full sort is safer.
        return np.argsort(dists_sq)
    

class ClusterSorter(BaseSorter):
    def __init__(self, n_clusters=100):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.clusters = None # Dict mapping cluster_id -> list of indices

    def fit(self, X):
        self.labels = self.kmeans.fit_predict(X)
        # Pre-group indices by cluster for fast retrieval
        self.argsort_clusters = np.argsort(self.labels)
        self.boundaries = np.searchsorted(self.labels[self.argsort_clusters], range(self.kmeans.n_clusters + 1))
        return self

    def get_sorted_indices(self, q, X=None):
        # 1. Find closest clusters to query
        cluster_dists = self.kmeans.transform(q.reshape(1, -1))[0]
        sorted_cluster_ids = np.argsort(cluster_dists)
        
        # 2. Concatenate indices from closest clusters first
        # This approximates a full sort but is O(C log C) instead of O(N log N)
        sorted_indices = []
        for cid in sorted_cluster_ids:
            start = self.boundaries[cid]
            end = self.boundaries[cid+1]
            sorted_indices.append(self.argsort_clusters[start:end])
            
        return np.concatenate(sorted_indices)