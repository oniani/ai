import numpy as np

from dataclasses import dataclass


@dataclass
class KNN:
    features: np.ndarray
    labels: np.ndarray
    k: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using the given features."""

        num_samples, _ = features.shape

        predictions = np.empty(num_samples)
        for idx, feature in enumerate(features):
            distances = [np.linalg.norm(feature - train_feature) for train_feature in self.features]
            k_sorted_idxs = np.argsort(distances)[: self.k]
            most_common = np.bincount([self.labels[idx] for idx in k_sorted_idxs]).argmax()
            predictions[idx] = most_common

        return predictions


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()

    train_features, test_features, train_labels, test_labels = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=0,
    )

    knn = KNN(train_features, train_labels, k=3)
    predictions = knn.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {recall:.3f}")
    print(f"Recall:    {precision:.3f}")
    print(f"F-score:   {fscore:.3f}")
