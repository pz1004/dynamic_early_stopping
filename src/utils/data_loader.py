"""
Dataset loading utilities for k-NN experiments.

Provides unified interface for loading:
- MNIST
- Fashion-MNIST
- CIFAR-10 (with feature extraction)
- Synthetic datasets
- GloVe word vectors
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import os


class DataLoader:
    """
    Unified data loader for all experiment datasets.

    Parameters
    ----------
    data_dir : str, default='./data'
        Directory to store/load datasets.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, data_dir: str = './data', random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def load(
        self,
        dataset_name: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset: 'mnist', 'fashion_mnist', 'cifar10',
            'synthetic_clustered', 'synthetic_uniform'.
        **kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        X_train : np.ndarray
            Training data (database to search).
        X_test : np.ndarray
            Test data (query points).
        y_train : np.ndarray
            Training labels (if available).
        y_test : np.ndarray
            Test labels (if available).
        """
        loaders = {
            'mnist': self._load_mnist,
            'fashion_mnist': self._load_fashion_mnist,
            'cifar10': self._load_cifar10,
            'cifar10_resnet': self._load_cifar10_features,
            'synthetic_clustered': self._load_synthetic_clustered,
            'synthetic_uniform': self._load_synthetic_uniform,
            'glove': self._load_glove,
        }

        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(loaders.keys())}")

        return loaders[dataset_name](**kwargs)

    def _load_mnist(
        self,
        normalize: bool = True,
        flatten: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load MNIST dataset.

        MNIST: 70,000 handwritten digits (28x28 grayscale images)
        - 60,000 training, 10,000 test
        - 784 dimensions when flattened
        """
        try:
            # Try sklearn first (easiest)
            from sklearn.datasets import fetch_openml

            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target.astype(np.int32)

        except Exception:
            try:
                # Try torchvision
                import torchvision
                import torchvision.transforms as transforms

                transform = transforms.ToTensor()

                train_dataset = torchvision.datasets.MNIST(
                    root=str(self.data_dir), train=True, download=False, transform=transform
                )
                test_dataset = torchvision.datasets.MNIST(
                    root=str(self.data_dir), train=False, download=False, transform=transform
                )

                X_train = train_dataset.data.numpy().reshape(-1, 784)
                X_test = test_dataset.data.numpy().reshape(-1, 784)
                y_train = train_dataset.targets.numpy()
                y_test = test_dataset.targets.numpy()

                X = np.vstack([X_train, X_test])
                y = np.concatenate([y_train, y_test])

            except Exception:
                # Generate synthetic MNIST-like data for testing
                print("Warning: Could not load MNIST. Using synthetic data.")
                return self._load_synthetic_uniform(n=70000, d=784)

        # Split
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        if flatten:
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)
        else:
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
        
        return X_train, X_test, y_train, y_test

    def _load_fashion_mnist(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load Fashion-MNIST dataset.

        Fashion-MNIST: 70,000 fashion items (28x28 grayscale images)
        - Same structure as MNIST
        - 10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
          Sandal, Shirt, Sneaker, Bag, Ankle boot
        """
        try:
            from sklearn.datasets import fetch_openml

            fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
            X, y = fmnist.data, fmnist.target.astype(np.int32)

        except Exception:
            try:
                import torchvision
                import torchvision.transforms as transforms

                transform = transforms.ToTensor()

                train_dataset = torchvision.datasets.FashionMNIST(
                    root=str(self.data_dir), train=True, download=False, transform=transform
                )
                test_dataset = torchvision.datasets.FashionMNIST(
                    root=str(self.data_dir), train=False, download=False, transform=transform
                )

                X_train = train_dataset.data.numpy().reshape(-1, 784)
                X_test = test_dataset.data.numpy().reshape(-1, 784)
                y_train = train_dataset.targets.numpy()
                y_test = test_dataset.targets.numpy()

                X = np.vstack([X_train, X_test])
                y = np.concatenate([y_train, y_test])

            except Exception:
                print("Warning: Could not load Fashion-MNIST. Using synthetic data.")
                return self._load_synthetic_uniform(n=70000, d=784)

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        return X_train, X_test, y_train, y_test

    def _load_cifar10(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 raw pixels.

        CIFAR-10: 60,000 color images (32x32x3)
        - 50,000 training, 10,000 test
        - 3072 dimensions when flattened
        """
        try:
            import torchvision
            import torchvision.transforms as transforms

            transform = transforms.ToTensor()

            train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True, download=False, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False, download=False, transform=transform
            )

            X_train = np.array([img.numpy().flatten() for img, _ in train_dataset])
            X_test = np.array([img.numpy().flatten() for img, _ in test_dataset])
            y_train = np.array(train_dataset.targets)
            y_test = np.array(test_dataset.targets)

        except Exception:
            print("Warning: Could not load CIFAR-10. Using synthetic data.")
            return self._load_synthetic_uniform(n=60000, d=3072)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        return X_train, X_test, y_train, y_test

    def _load_cifar10_features(
        self,
        model_name: str = 'resnet18'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 with features extracted from pretrained ResNet.

        Parameters
        ----------
        model_name : str
            Model to use: 'resnet18' (512 dims) or 'resnet50' (2048 dims).
        """
        cache_path = self.data_dir / f'cifar10_{model_name}_features.npz'

        if cache_path.exists():
            # Load from cache
            data = np.load(cache_path)
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']

        try:
            import torch
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader

            # Setup transforms (ResNet expects 224x224 but we'll resize)
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True, download=False, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False, download=False, transform=transform
            )

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            # Load pretrained model
            if model_name == 'resnet18':
                model = torchvision.models.resnet18(pretrained=True)
                feature_dim = 512
            else:
                model = torchvision.models.resnet50(pretrained=True)
                feature_dim = 2048

            # Remove final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Extract features
            def extract_features(loader):
                features = []
                labels = []
                with torch.no_grad():
                    for images, targets in loader:
                        images = images.to(device)
                        feats = model(images).squeeze()
                        features.append(feats.cpu().numpy())
                        labels.append(targets.numpy())
                return np.vstack(features), np.concatenate(labels)

            X_train, y_train = extract_features(train_loader)
            X_test, y_test = extract_features(test_loader)

            # Cache for future use
            np.savez(cache_path, X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

        except Exception as e:
            print(f"Warning: Could not extract CIFAR-10 features: {e}")
            print("Using synthetic data.")
            feature_dim = 512 if model_name == 'resnet18' else 2048
            return self._load_synthetic_uniform(n=60000, d=feature_dim)

        return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test

    def _load_synthetic_clustered(
        self,
        n: int = 100000,
        d: int = 128,
        n_clusters: int = 50,
        cluster_std: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic clustered data using Gaussian mixture.

        Parameters
        ----------
        n : int
            Total number of samples.
        d : int
            Dimensionality.
        n_clusters : int
            Number of Gaussian clusters.
        cluster_std : float
            Standard deviation within clusters.
        """
        # Generate cluster centers
        centers = self.rng.standard_normal((n_clusters, d)) * 10

        # Assign points to clusters
        labels = self.rng.integers(0, n_clusters, n)

        # Generate points around cluster centers
        X = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            X[i] = centers[labels[i]] + self.rng.standard_normal(d) * cluster_std

        # Split 90-10
        n_train = int(0.9 * n)
        perm = self.rng.permutation(n)

        X_train = X[perm[:n_train]]
        X_test = X[perm[n_train:]]
        y_train = labels[perm[:n_train]]
        y_test = labels[perm[n_train:]]

        return X_train, X_test, y_train, y_test

    def _load_synthetic_uniform(
        self,
        n: int = 100000,
        d: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic uniformly distributed data.

        This represents "hard" case where there's no clustering structure.
        """
        X = self.rng.standard_normal((n, d)).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)  # No meaningful labels

        n_train = int(0.9 * n)

        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

    def _load_glove(
        self,
        dim: int = 50,
        max_words: int = 400000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load GloVe word vectors.

        Parameters
        ----------
        dim : int
            Embedding dimension (50, 100, 200, or 300).
        max_words : int
            Maximum number of words to load.
        """
        glove_path = self.data_dir / f'glove.6B.{dim}d.txt'

        if not glove_path.exists():
            print(f"Warning: GloVe file not found at {glove_path}")
            print("Using synthetic data.")
            return self._load_synthetic_uniform(n=max_words, d=dim)

        vectors = []
        words = []

        with open(glove_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vector) == dim:
                    words.append(word)
                    vectors.append(vector)

        X = np.array(vectors)
        y = np.arange(len(vectors))  # Word indices as labels

        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-10)

        n_train = int(0.9 * len(X))

        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata about a dataset."""
        info = {
            'mnist': {
                'n_train': 60000,
                'n_test': 10000,
                'dims': 784,
                'n_classes': 10,
                'description': 'Handwritten digits 0-9'
            },
            'fashion_mnist': {
                'n_train': 60000,
                'n_test': 10000,
                'dims': 784,
                'n_classes': 10,
                'description': 'Fashion items (clothing, shoes, bags)'
            },
            'cifar10': {
                'n_train': 50000,
                'n_test': 10000,
                'dims': 3072,
                'n_classes': 10,
                'description': 'Color images of objects'
            },
            'cifar10_resnet': {
                'n_train': 50000,
                'n_test': 10000,
                'dims': 512,
                'n_classes': 10,
                'description': 'CIFAR-10 with ResNet18 features'
            },
            'synthetic_clustered': {
                'n_train': 90000,
                'n_test': 10000,
                'dims': 128,
                'n_classes': 50,
                'description': 'Gaussian mixture clusters'
            },
            'synthetic_uniform': {
                'n_train': 90000,
                'n_test': 10000,
                'dims': 128,
                'n_classes': 0,
                'description': 'Uniform random data (hard case)'
            }
        }
        return info.get(dataset_name, {})
