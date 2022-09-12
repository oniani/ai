import numpy as np

from dataclasses import dataclass


@dataclass
class GaussianNaiveBayes:
    features: np.ndarray
    labels: np.ndarray

    def fit(self) -> None:
        """Fits the Gaussian Naive Bayes model."""

        self.unique_labels = np.unique(self.labels)

        self.params = []
        # For the given label, calculate the mean and variance of all features
        for label in self.unique_labels:
            label_features = self.features[self.labels == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])

    def likelihood(self, data: float, mean: float, var: float) -> float:
        """Calculates the Gaussian likelihood of the data with the given mean and variance."""

        # NOTE: Added in denominator to prevent division by zero
        eps = 1e-4

        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((data - mean) ** 2 / (2 * var + eps)))

        return coeff * exponent

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using Bayes' Theorem:  P(A | B) = P(B | A) * P(A) / P(B)."""

        num_samples, _ = features.shape

        predictions = np.empty(num_samples)
        for idx, feature in enumerate(features):
            posteriors = []
            for label_idx, label in enumerate(self.unique_labels):
                # Prior is the mean of what we have
                prior = (self.labels == label).mean()

                # Naive assumption (independence):
                #   P(a0, a1, a2 | B) = P(a0 | B) * P(a1 | B) * P(a2 | B)
                likelihood = np.prod(
                    [self.likelihood(f, m, v) for f, (m, v) in zip(feature, self.params[label_idx])]
                )

                # Posterior = Prior * Likelihood / Scaling Factor (ignoring scaling factor)
                posteriors.append(prior * likelihood)

            # Store the label with the largest posterior probability
            predictions[idx] = self.unique_labels[np.argmax(posteriors)]

        return predictions


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    features, labels = load_iris(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.5, random_state=0
    )

    gnb = GaussianNaiveBayes(train_features, train_labels)  # type: ignore
    gnb.fit()
    predictions = gnb.predict(test_features)  # type: ignore

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F-score:   {fscore:.3f}")
    print()
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")  # type: ignore
