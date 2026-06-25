import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


MNIST_URLS = {
    "train-images-idx3-ubyte.gz": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    ],
    "train-labels-idx1-ubyte.gz": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    ],
}


def download_file(file_name: str, save_dir: Path):
    save_path = save_dir / file_name

    if save_path.exists():
        print(f"Already exists: {save_path}")
        return save_path

    last_error = None

    for url in MNIST_URLS[file_name]:
        try:
            print(f"Downloading {file_name} from {url}")
            urllib.request.urlretrieve(url, save_path)
            print(f"Saved: {save_path}")
            return save_path
        except Exception as e:
            last_error = e
            print(f"Failed: {url}")
            print(f"Reason: {e}")

    raise RuntimeError(f"Cannot download {file_name}. Last error: {last_error}")


def read_mnist_first_n(image_path: Path, label_path: Path, limit=20):
    images = []
    labels = []

    with gzip.open(image_path, "rb") as image_file, gzip.open(label_path, "rb") as label_file:
        image_magic, num_images, num_rows, num_cols = struct.unpack(">IIII", image_file.read(16))
        label_magic, num_labels = struct.unpack(">II", label_file.read(8))

        if image_magic != 2051:
            raise ValueError("Invalid MNIST image file")
        if label_magic != 2049:
            raise ValueError("Invalid MNIST label file")
        if num_images != num_labels:
            raise ValueError("Image count and label count do not match")

        read_count = min(limit, num_images)

        for _ in range(read_count):
            image_bytes = image_file.read(num_rows * num_cols)
            label_byte = label_file.read(1)

            image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(num_rows, num_cols)
            label = int.from_bytes(label_byte, byteorder="big")

            images.append(image)
            labels.append(label)

    return images, labels


def export_png(images, labels, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image, mode="L")

        file_name = f"mnist_{i:02d}_label_{label}.png"
        save_path = output_dir / file_name

        img.save(save_path)
        print(f"Exported: {save_path}")


def main():
    dataset_dir = Path("mnist_raw")
    output_dir = Path("")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_images_path = "../data_set/training_set/train-images-idx3-ubyte.gz"
    train_labels_path = "../data_set/training_set/train-labels-idx1-ubyte.gz"

    images, labels = read_mnist_first_n(
        train_images_path,
        train_labels_path,
        limit=20
    )

    export_png(images, labels, output_dir)

    print("\nDone.")
    print(f"Raw MNIST files saved in: {dataset_dir}")
    print(f"20 PNG images saved in: {output_dir}")


if __name__ == "__main__":
    main()
