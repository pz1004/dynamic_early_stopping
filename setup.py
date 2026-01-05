from setuptools import setup, find_packages

setup(
    name='des_knn',
    version='0.1.0',
    description='Dynamic Early Stopping for k-Nearest Neighbors Search',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'pyyaml>=6.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
        ],
        'deep': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
        ],
        'ann': [
            'annoy>=1.17.0',
            'hnswlib>=0.6.0',
        ]
    }
)
