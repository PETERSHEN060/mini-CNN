import gzip
import struct
import numpy as np

from Data_Sturcture import DataMatrix

def open_mnist_file(file_path):
    """
    打开 MNIST 文件。
    如果是 .gz，就用 gzip 打开。
    如果不是 .gz，就按普通二进制文件打开。
    """
    if file_path.endswith(".gz"):
        return gzip.open(file_path, "rb")
    else:
        return open(file_path, "rb")


def read_mnist_dataset(image_file_path,
                       label_file_path,
                       limit=None):
    """
    读取 MNIST 图片文件和标签文件，
    并把每一张图片存成 PicutreMatrix 对象。

    image_file_path:
        例如 train-images-idx3-ubyte.gz

    label_file_path:
        例如 train-labels-idx1-ubyte.gz

    limit:
        只读取前多少张图片。
        如果是 None，就读取全部。

    返回：
        list[PicutreMatrix]
    """

    dataset = []

    with open_mnist_file(image_file_path) as image_file, open_mnist_file(label_file_path) as label_file:
        # 读取图片文件头部
        # 图片文件头格式：
        # magic number: 4 bytes
        # number of images: 4 bytes
        # number of rows: 4 bytes
        # number of columns: 4 bytes
        image_magic, num_images, num_rows, num_cols = struct.unpack(">IIII", image_file.read(16))

        # 读取标签文件头部
        # 标签文件头格式：
        # magic number: 4 bytes
        # number of labels: 4 bytes
        label_magic, num_labels = struct.unpack(">II", label_file.read(8))

        # 检查 magic number
        if image_magic != 2051:
            raise ValueError("图片文件 magic number 错误，不是合法的 MNIST image 文件")

        if label_magic != 2049:
            raise ValueError("标签文件 magic number 错误，不是合法的 MNIST label 文件")

        # 检查图片和标签数量是否一致
        if num_images != num_labels:
            raise ValueError("图片数量和标签数量不一致")

        # 检查图片大小
        if num_rows != 28 or num_cols != 28:
            raise ValueError(f"图片大小错误，期望 28x28，但得到 {num_rows}x{num_cols}")

        # 决定读取多少张
        if limit is None:
            read_count = num_images
        else:
            read_count = min(limit, num_images)

        for _ in range(read_count):
            # 一张图片有 28 * 28 = 784 个像素
            image_bytes = image_file.read(28 * 28)

            # 一个标签是 1 byte
            label_byte = label_file.read(1)

            # 把二进制像素转成 uint8 数组
            image = np.frombuffer(image_bytes, dtype=np.uint8)

            # 改成 28 x 28
            image = image.reshape(28, 28)

            # 归一化到 0.0 ~ 1.0，并转成 float32
            image = image.astype(np.float32) / 255.0

            # 转成 1 x 28 x 28
            image = image.reshape(1, 28, 28)

            # 标签转成整数
            label = int.from_bytes(label_byte, byteorder="big")

            # 存成 PicutreMatrix 对象
            picture = DataMatrix(image, label)

            dataset.append(picture)

    return dataset


if __name__ == "__main__":
    train_images_path = "train-images-idx3-ubyte.gz"
    train_labels_path = "train-labels-idx1-ubyte.gz"

    # 先读取前 5 张测试一下
    train_dataset = read_mnist_dataset(
        train_images_path,
        train_labels_path,
        limit=5
    )

    print("读取完成，数据数量：", len(train_dataset))

    first_picture = train_dataset[0]

    first_picture.show_info()

    print("\n第一张图片的字符显示：")
    first_picture.show_ascii()
