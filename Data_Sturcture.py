import numpy as np


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

    def flatten(self):
        """
        把image拉成 1 x N X N
        """
        return self.image.reshape(-1)

    def show_info(self):
        """
        打印图片基本信息。
        """
        print("Image shape:", self.image.shape)
        print("Image dtype:", self.image.dtype)
        print("Label:", self.label)
        print("Min pixel:", self.image.min())
        print("Max pixel:", self.image.max())

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
        return self.feature[channel, row, col]

    def set_value(self, channel, row, col, value):
        self.feature[channel, row, col] = np.float32(value)

    def flatten(self):
        return self.feature.reshape(-1)

    def copy(self):
        return FeatureMap(self.feature.copy(), self.label)

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

class Vector:
    """
    用来存储 flatten 后的一维向量。
    """

    def __init__(self, vector_data, label=None):
        vector = np.array(vector_data, dtype=np.float32)

        # ----输入检查----
        if vector.ndim != 1:
            raise ValueError(f"Vector 维度错误，期望是一维，但得到 {vector.shape}")
        if vector.shape[0] <= 0:
            raise ValueError("Vector 长度必须大于 0")
        if label is not None and not isinstance(label, int):
            raise TypeError("label 必须是整数类别编号，或者为 None")

        self.vector = vector
        self.label = label
        self.length = vector.shape[0]

    def get_value(self, index):
        return self.vector[index]

    def set_value(self, index, value):
        self.vector[index] = np.float32(value)

    def copy(self):
        return Vector(self.vector.copy(), self.label)

    def show_info(self):
        """
        打印 Vector 基本信息。
        """
        print("Vector shape:", self.vector.shape)
        print("Vector dtype:", self.vector.dtype)
        print("Vector length:", self.length)
        print("Label:", self.label)
        print("Min value:", self.vector.min())
        print("Max value:", self.vector.max())
        print("Mean value:", self.vector.mean())

class Kernel:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        # 随机初始化，n*n w矩阵, float32降低存储，
        # 0.01降低权重防止反向传播出问题
        self.weights = np.random.randn(kernel_size, kernel_size).astype(np.float32) * 0.01
