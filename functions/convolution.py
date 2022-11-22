import numpy as np


def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Computes a 2D Convolution of a single-channel image via given kernel, stride, and padding."""

    # Flipping a kernel is necessary as otherwise, we get cross-correlation, which is a related, but
    # different operation. Convolution has some nice mathematical properties, which are not always
    # present in cross-correlation.
    kernel = np.flipud(np.fliplr(kernel))

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    out_height = int(np.floor((image_height - kernel_height + 2 * padding) / stride) + 1)
    out_width = int(np.floor((image_width - kernel_width + 2 * padding) / stride) + 1)

    out = np.zeros((out_height, out_width))
    for h in range(out_height):
        for w in range(out_width):
            out[h][w] = (image[h : h + kernel_height, w : w + kernel_width] * kernel).sum()

    return out


if __name__ == "__main__":
    import matplotlib.image as image
    import matplotlib.pyplot as plt

    def rgb2grayscale(image: np.ndarray) -> np.ndarray:
        """Converts an RGB image to a grayscale image."""

        return image[..., :3].dot([0.2989, 0.5870, 0.1140])

    # Ridge detection
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Gaussian blur (3 x 3)
    # kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    original = image.imread("baby_yoda.jpg")
    grayscale = rgb2grayscale(original)
    plots = {
        "Original": (original, "darkblue"),
        "Grayscale": (grayscale, "gray"),
        "Post-Convolution": (conv2d(grayscale, kernel), "orangered"),
    }

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for idx, (key, (img, c)) in enumerate(plots.items()):
        ax = axs[idx]
        ax.imshow(img)
        ax.set_title(key, c=c)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("new_baby_yoda.jpg", bbox_inches="tight", pad_inches=0, dpi=256)
