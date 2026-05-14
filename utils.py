from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler

CWD = Path(".")
YES_PATH = CWD / "data" / "yes"
NO_PATH = CWD / "data" / "no"
DATASETS_DIR = CWD / "datasets"
SIZE = 128


def load_image(path):
    img = Image.open(path).convert("L")
    # image non étirée (bandes noires)
    img_padded = ImageOps.pad(
        img, (SIZE, SIZE), method=Image.Resampling.LANCZOS, color=0
    )

    return np.array(img_padded)


def plot_image(arr):
    plt.imshow(arr, cmap="gray")
    plt.show()


class Dataset:
    def __init__(self, name) -> None:
        self.base_path = DATASETS_DIR / name
        self.s = StandardScaler()
        self.train = self.load_train()
        self.test = self.load_test()

    def load_train(self):
        all_images = []
        all_labels = []

        path = self.base_path / "train"
        self.mapping = {}
        for i, category in enumerate(path.iterdir()):
            images = [load_image(p) for p in category.iterdir()]
            labels = [i] * len(images)

            all_images.extend(images)
            all_labels.extend(labels)
            self.mapping[i] = category.stem

        X = np.array(all_images)
        y = np.array(all_labels)

        self.num_classes = i + 1
        idx = np.random.permutation(len(X))

        X = X[idx].reshape((-1, SIZE * SIZE))
        y = y[idx]

        X = self.s.fit_transform(X)
        X = X.reshape((-1, SIZE, SIZE, 1))

        return X, y

    def load_test(self):
        all_images = []
        all_labels = []

        path = self.base_path / "test"
        for i, category in enumerate(path.iterdir()):
            images = [load_image(p) for p in category.iterdir()]
            labels = [i] * len(images)

            all_images.extend(images)
            all_labels.extend(labels)

        X = np.array(all_images).reshape((-1, SIZE * SIZE))
        y = np.array(all_labels)

        idx = np.random.permutation(len(X))

        X = X[idx].reshape((-1, SIZE * SIZE))
        y = y[idx]

        X = self.s.transform(X)
        X = X.reshape((-1, SIZE, SIZE, 1))

        return X, y

    def reverse_transform(self, img):
        flat = img.reshape((1, -1))
        raw = self.s.inverse_transform(flat)
        return raw.reshape((SIZE, SIZE))
