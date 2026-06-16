import numpy as np
from numpy.distutils.fcompiler import none


class DataMatrix:
    """
    用来存储一张灰度图矩阵。

    形状：1 x H x W
    像素范围：0.0 ~ 1.0
    数据类型：float32
    label 可以有，也可以没有
    """

    def __init__(self, image_data, label=None):
        image = np.array(image_data, dtype=np.float32)

        # ----输入检查----
        # 如果是 H x W，改成 1 x H x W
        if image.ndim == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])
        # 检查必须是 1 x H x W
        if image.ndim != 3:
            raise ValueError(f"图片维度错误，期望是 3 维，但得到 {image.shape}")
        if image.shape[0] != 1:
            raise ValueError(f"当前只支持灰度图，期望 channel=1，但得到 {image.shape[0]}")
        # 检查像素范围
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("像素值必须在 0.0 到 1.0 之间")
        # 检查标签
        if label is not None and not isinstance(label, int):
            raise TypeError("label 必须是整数类别编号，或者为 None")

        # ----赋值----
        self.image = image
        self.label = label
        self.channel = image.shape[0]
        self.height = image.shape[1]
        self.width = image.shape[2]

    def get_pixel(self, row, col):
        """
        获取某个像素。
        row: 0~27
        col: 0~27
        """
        return self.image[0, row, col]

    def set_pixel(self, row, col, value):
        """
        设置某个像素。
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("像素值必须在 0.0 到 1.0 之间")

        self.image[0, row, col] = np.float32(value)

    # def flatten(self):
    #     """
    #     把 1 x 28 x 28 拉平成 784 维向量。
    #     """
    #     return self.image.reshape(784)

    def show_info(self):
        """
        打印图片基本信息。
        """
        print("Image shape:", self.image.shape)
        print("Image dtype:", self.image.dtype)
        print("Label:", self.label)
        print("Min pixel:", self.image.min())
        print("Max pixel:", self.image.max())

    # def show_ascii(self):
    #     """
    #     用字符简单显示图片。
    #     数值越大，字符越明显。
    #     """
    #     chars = " .:-=+*#%@"
    #
    #     for row in range(28):
    #         line = ""
    #         for col in range(28):
    #             pixel = self.image[0, row, col]
    #             index = int(pixel * (len(chars) - 1))
    #             line += chars[index]
    #         print(line)

class FeatureMap:
    """
    用来存储卷积层输出的特征。

    形状：C x H x W
    数据类型：float32
    DataMatrix 存原始图片，像素范围通常是 0.0 ~ 1.0。
    FeatureMap 存卷积结果，数值可以小于 0，也可以大于 1。
    """

    def __init__(self, feature_data, label=None):
        # 复制input
        feature = np.array(feature_data, dtype=np.float32)

        # ----输入检查----
        # 如果是 H x W，改成 1 x H x W
        if feature.ndim == 2:
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        # 检查必须是 C x H x W
        if feature.ndim != 3:
            raise ValueError(f"FeatureMap 维度错误，期望是 3 维，但得到 {feature.shape}")
        if feature.shape[0] <= 0:
            raise ValueError("FeatureMap 的 channel 数必须大于 0")

        # ----赋值----
        self.feature = feature
        self.label = label
        self.channel = feature.shape[0]
        self.height = feature.shape[1]
        self.width = feature.shape[2]

    def get_value(self, channel, row, col):
        """
        获取某个 channel、某个位置的特征值。
        """
        return self.feature[channel, row, col]

    def set_value(self, channel, row, col, value):
        """
        设置某个 channel、某个位置的特征值。
        """
        self.feature[channel, row, col] = np.float32(value)

    def flatten(self):
        """
        把 C x H x W 拉平成一维向量。
        """
        return self.feature.reshape(-1)

    def copy(self):
        """
        返回一个新的 FeatureMap，避免直接修改原对象。
        """
        return FeatureMap(self.feature.copy(), self.label)

    def relu(self):
        """
        对 FeatureMap 做 ReLU，返回新的 FeatureMap。
        不修改原 FeatureMap。
        """
        relu_feature = np.maximum(0, self.feature)
        return FeatureMap(relu_feature, self.label)

    def show_info(self):
        """
        打印 FeatureMap 基本信息。
        """
        print("FeatureMap shape:", self.feature.shape)
        print("FeatureMap dtype:", self.feature.dtype)
        print("Label:", self.label)
        print("Channel:", self.channel)
        print("Height:", self.height)
        print("Width:", self.width)
        print("Min value:", self.feature.min())
        print("Max value:", self.feature.max())
        print("Mean value:", self.feature.mean())

class Kernel:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

        self.weights = np.random.randn(
            kernel_size,
            kernel_size
        ).astype(np.float32) * 0.01 # 随机初始化，n*n w矩阵, float32降低存储，
        # 0.01降低权重防止反向传播出问题
