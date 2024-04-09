#!/usr/bin/env python
import numpy as np


def pca(
    data: np.ndarray,
    n_components: int,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs dimensionality reduction via Principal Component Analysis."""

    if standardize:
        data = (data - data.mean(axis=0)) / data.std(axis=0)

    cov = np.cov(data.T)
    eigval, eigvec = np.linalg.eigh(cov)

    # idx = np.argsort(eigval)[::-1][:n_components]
    idx = np.arange(len(eigval) - 1, len(eigval) - 1 - n_components, -1)
    eigval, eigvec = eigval[idx], eigvec[:, idx]

    feat = data @ eigvec

    return feat, eigval, eigvec


# Eval {{{

if __name__ == "__main__":
    np.random.seed(123)

    n_components = 3
    data = np.random.rand(3, 4)

    # scikit-learn {{{

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    sklearn_pca = PCA(n_components=n_components)
    out = sklearn_pca.fit_transform(StandardScaler().fit_transform(data))

    # }}}

    # NumPy {{{

    feat, eigval, eigvec = pca(data=data, n_components=n_components)

    # }}}

    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    reconstructed = feat @ eigvec.T

    print(np.allclose(out, feat))
    print(np.allclose(standardized, reconstructed))

# }}}
