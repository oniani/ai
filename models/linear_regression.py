import numpy as np

from dataclasses import dataclass


@dataclass
class LinearRegression:
    learning_rate: float
    epochs: int
    logging: bool

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the Linear Regression model."""

        num_samples, num_features = features.shape
        self.weights, self.bias = np.zeros(num_features), 0

        for epoch in range(self.epochs):
            residuals = labels - self.predict(features)  # type: ignore

            d_weights = -2 / num_samples * residuals.dot(features)
            d_bias = -2 / num_samples * residuals.sum()

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            if self.logging:
                print(f"MSE Loss [{epoch}]: {(residuals ** 2).mean():.3f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""

        return features.dot(self.weights) + self.bias


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    plt.style.use("bmh")

    # Training dataset
    train_features = np.arange(0, 250).reshape(-1, 1)
    train_labels = np.arange(0, 500, 2)

    # Testing dataset
    test_features = np.arange(300, 400, 8).reshape(-1, 1)
    test_labels = np.arange(600, 800, 16)

    linear_regression = LinearRegression(epochs=25, learning_rate=1e-5, logging=False)
    linear_regression.fit(train_features, train_labels)
    predictions = linear_regression.predict(test_features).round()

    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("f(x) = 2x")
    fig.tight_layout()
    fig.set_size_inches(18, 8)

    axs[0].set_title("Visualization for f(x) = 2x")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].plot(train_features, train_labels)

    axs[1].set_title("Scatterplot for f(x) = 2x Data")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].scatter(test_features, test_labels, color="blue")

    axs[2].set_title("Visualization for Approximated f(x) = 2x")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].scatter(test_features, test_labels, color="blue")
    axs[2].plot(test_features, predictions)

    plt.show()

    accuracy = accuracy_score(predictions, test_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {recall:.3f}")
    print(f"Recall:    {precision:.3f}")
    print(f"F-score:   {fscore:.3f}")
