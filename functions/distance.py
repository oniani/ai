import numpy as np


def euclidean_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Euclidean distance between two tensors."""

    return np.linalg.norm(t1 - t2)


def manhattan_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Manhattan distance between two tensors."""

    return np.abs(t1 - t2).sum()


def cosine_distance(t1: np.ndarray, t2: np.ndarray) -> np.float64:
    """Computes the Cosine distance between two tensors."""

    if np.count_nonzero(t1) == 0 or np.count_nonzero(t2) == 0:
        raise ValueError("Cosine distance is undefined for zero-tensors")

    return t1.dot(t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))


if __name__ == "__main__":
    np.random.seed(0)

    t1 = np.random.rand(4)
    # t1 /= np.linalg.norm(t1)

    t2 = np.random.rand(4)
    # t2 /= np.linalg.norm(t2)

    # t1 = np.array([3, 4])
    # t2 = np.array([0, 0])

    print(t1)
    print(t2)
    print("----------------------------")
    print("      Distance Metrics")
    print("----------------------------")
    print(f"- Euclidean Distance: {euclidean_distance(t1, t2):.4f}")
    print(f"- Manhattan Distance: {manhattan_distance(t1, t2):.4f}")
    print(f"- Cosine    Distance: {cosine_distance(t1, t2):.4f}")
