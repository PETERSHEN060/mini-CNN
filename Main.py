import numpy as np
from PIL import Image

from Data_Sturcture import DataMatrix
from CNN_Networks import CNNNetwork


def read_28x28_image(image_path, invert=True) -> DataMatrix:
    """
    读取一张 28x28 图片，并转换成 DataMatrix。

    参数：
        image_path:
            图片路径，例如 "my_digit.png"

        invert:
            是否反色。

            MNIST 是黑底白字。
            如果你的图片是白底黑字，应该 invert=True。
            如果你的图片本来就是黑底白字，应该 invert=False。

    返回：
        DataMatrix，shape = 1 x 28 x 28
    """

    # 1. 打开图片并转成灰度图
    image = Image.open(image_path).convert("L")

    # 2. 如果不是 28x28，强制 resize 成 28x28
    image = image.resize((28, 28))

    # 3. 转成 numpy array
    image_array = np.array(image, dtype=np.float32)

    # 4. 归一化到 0.0 ~ 1.0
    image_array = image_array / 255.0

    # 5. 如果是白底黑字，反色成 MNIST 风格：黑底白字
    if invert:
        image_array = 1.0 - image_array

    # 6. 转成 1 x 28 x 28
    image_array = image_array.reshape(1, 28, 28)

    # 7. label=None，因为预测图片没有真实标签
    return DataMatrix(image_array, label=None)


def predict_custom_image(image_path, weights_path="model/cnn_weight1.npz"):
    """
    读取本地 28x28 图片，并使用训练好的权重预测。
    """

    # 1. 读取图片
    data = read_28x28_image(
        image_path=image_path,
        invert=True
    )

    # 2. 创建模型
    model = CNNNetwork()

    # 3. 读取训练好的权重
    model.load_weights(weights_path)

    # 4. 预测
    prediction = model.predict_one(data)

    print("图片路径:", image_path)
    print("预测结果:", prediction)


if __name__ == "__main__":
    image_path = "mnist_20_png/mnist_08_label_1.png"
    weights_path = "model/cnn_weight1.npz"
    predict_custom_image(image_path, weights_path)
