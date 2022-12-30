import numpy as np

from dataclasses import dataclass


@dataclass
class KMeans:
    k: int
    iterations: int
    tol: float

    def fit(self, features: np.ndarray) -> None:
        """Clusters the data."""

        num_samples, _ = features.shape

        self.centroids = features[np.random.choice(num_samples, size=self.k, replace=False)]
        self.closest = np.zeros(num_samples)

        for _ in range(self.iterations):
            old_closest = self.closest.copy()

            distances = [np.linalg.norm(self.centroids - feature, axis=1) for feature in features]
            self.closest = np.argmin(distances, axis=1)

            for idx in range(self.k):
                self.centroids[idx] = (features[self.closest == idx]).mean(axis=0)

            if np.linalg.norm(self.closest - old_closest) < self.tol:
                break


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 256,
            "figure.figsize": (1 << 4, 1 << 3),
            "font.family": "JetBrainsMono Nerd Font Mono",
            "font.size": 1 << 3,
        }
    )
    plt.style.use("bmh")

    np.random.seed(0)

    features = np.random.rand(1_000, 2)

    kmeans = KMeans(k=4, iterations=16, tol=1e-4)
    kmeans.fit(features)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("K-Means Clustering", fontsize=24)

    axs[0].scatter(features[:, 0], features[:, 1])
    axs[0].set_title("Before Clutering", fontsize=20)

    axs[1].scatter(features[:, 0], features[:, 1], c=kmeans.closest)
    axs[1].set_title("After Clutering", fontsize=20)

    plt.savefig("thumbnail.png")
