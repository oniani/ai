import numpy as np

from dataclasses import dataclass


@dataclass
class LogisticRegression:
    learning_rate: float
    epochs: int
    threshold: float
    logging: bool

    def sigmoid(self, predictions: np.ndarray) -> np.ndarray:
        """The numerically stable implementation of the Sigmoid activation function."""

        neg_mask = predictions < 1
        pos_mask = ~neg_mask

        zs = np.empty_like(predictions)
        zs[neg_mask] = np.exp(predictions[neg_mask])
        zs[pos_mask] = np.exp(-predictions[pos_mask])

        res = np.ones_like(predictions)
        res[neg_mask] = zs[neg_mask]

        return res / (1 + zs)

    def mean_log_loss(self, predictions: np.ndarray, labels: np.ndarray) -> np.float32:
        """Computes the mean Cross Entropy Loss (in binary classification, also called Log-loss)."""

        return -(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)).mean()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the Logistic Regression model."""

        num_samples, num_features = features.shape
        self.weights, self.bias = np.zeros(num_features), 0

        for epoch in range(self.epochs):
            prediction = self.sigmoid(features.dot(self.weights) + self.bias)
            difference = prediction - labels  # type: ignore

            d_weights = features.T.dot(difference) / num_samples
            d_bias = difference.sum() / num_samples

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            if self.logging:
                print(f"Mean Log-loss [{epoch}]: {self.mean_log_loss(prediction, labels):.3f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""

        return np.where(self.sigmoid(features.dot(self.weights) + self.bias) < self.threshold, 0, 1)  # type: ignore


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    plt.style.use("bmh")

    # Prepare the data
    data = load_breast_cancer()

    # Train/test split
    train_features, test_features, train_labels, test_labels = train_test_split(
        data.data, data.target, test_size=0.33, random_state=0  # type: ignore
    )

    logistic_regression = LogisticRegression(
        learning_rate=2e-5,
        epochs=256,
        threshold=0.5,
        logging=False,
    )
    logistic_regression.fit(train_features, train_labels)  # type: ignore
    predictions = logistic_regression.predict(test_features)  # type: ignore

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F-score:   {fscore:.3f}")
