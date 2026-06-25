import numpy as np
from Data_Sturcture import FeatureMap, Vector


class MaxPoollingLayer:
    def __init__(self, feature_map: FeatureMap, pool_size=2, stride=2):
        self.input = feature_map

        # 输入检查
        if pool_size <= 0:
            raise ValueError("pool_size 必须大于 0")
        if stride <= 0:
            raise ValueError("stride 必须大于 0")

        self.pool_size = pool_size
        self.stride = stride

        self.out_pooling = None
        self.out_flatten = None

        # backward
        self.input_shape = None
        self.pooling_output_shape = None
        # 回传时取的最大值的位置
        self.max_index_mask = None

    def max_pooling_forward(self) -> FeatureMap:
        """
        执行 Max Pooling。

        输入：
            self.input.feature: C x H x W

        输出：
            FeatureMap: C x output_H x output_W
        """

        # ----赋值----
        input_feature = self.input.feature
        channel = input_feature.shape[0]
        height = input_feature.shape[1]
        width = input_feature.shape[2]
        pool_size = self.pool_size
        stride = self.stride
        self.input_shape = input_feature.shape


        # ----计算输出尺寸----
        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("池化输出尺寸错误，请检查 pool_size 和 stride 是否合理")

        # 创建输出 FeatureMap
        output = np.zeros((channel, output_height, output_width), dtype=np.float32)

        # 用来记录最大值位置的镜像矩阵
        # shape 和 input_feature 一样
        self.max_index_mask = np.zeros_like(input_feature, dtype=np.float32)

        # ----执行 pooling----
        # 遍历图像的feature
        for c in range(channel):
            for out_row in range(output_height):
                for out_col in range(output_width):
                    # 定位初始位置
                    start_row = out_row * stride
                    start_col = out_col * stride
                    # 池化扫描的面积
                    region = input_feature[
                        c,
                        start_row:start_row + pool_size,
                        start_col:start_col + pool_size
                    ]
                    # 找到并输出最大值
                    max_value = np.max(region)
                    output[c, out_row, out_col] = max_value

                    # 找到 region 里最大值的位置，np.argmax 返回的是一维的index位置
                    max_index = np.argmax(region)
                    # unravel_index 找到最大值的实际2维坐标
                    max_row, max_col = np.unravel_index(max_index, region.shape)
                    # 把最大值位置记录到 mask 矩阵里
                    self.max_index_mask[c, start_row + max_row,start_col + max_col] = 1.0
        # 创建并保存FeatureMap
        self.out_pooling = FeatureMap(output, self.input.label)
        self.pooling_output_shape = output.shape

        return self.out_pooling


    def flatten_forward(self) -> Vector:
        """
        把池化后的 FeatureMap 拉平成 Vector。

        输入：
            FeatureMap.feature: C x H x W

        输出：
            Vector.vector: C * H * W
        """
        # 没有池化先池化
        if self.out_pooling is None:
            self.max_pooling_forward()

        # 拉平成1维
        feature = self.out_pooling.feature
        flatten_vector = feature.reshape(-1)

        # 保存并返回成vector类
        self.out_flatten = Vector(flatten_vector, self.input.label)
        return self.out_flatten


    def flatten_backward(self, d_vector: Vector) -> FeatureMap:
        """
        Flatten backward。

        输入：
            d_vector: Vector, shape = (C * H * W,)

        输出：
            FeatureMap, shape = C x H x W

        作用：
            把全连接层传回来的 1D 梯度，还原成池化输出的 3D 形状。
        """
        # 输入检查
        if not isinstance(d_vector, Vector):
            raise TypeError("flatten_backward 的输入必须是 Vector")
        if self.out_pooling is None:
            raise ValueError("必须先执行 max_pooling 或 flatten，才能做 flatten_backward")

        # 计算目标矩阵格式
        target_shape = self.out_pooling.feature.shape
        if d_vector.length != np.prod(target_shape):
            raise ValueError(f"梯度长度错误，期望 {np.prod(target_shape)}，但得到 {d_vector.length}")

        # 转换并返回成目标矩阵格式
        d_feature = d_vector.vector.reshape(target_shape)
        return FeatureMap(d_feature, self.input.label)


    def max_pooling_backward(self, d_pool_output: FeatureMap) -> FeatureMap:
        """
        Max Pooling backward。

        输入：
            d_pool_output: FeatureMap
                shape = pooling 输出的 C x output_H x output_W

        输出：
            d_input: FeatureMap
                shape = pooling 输入的 C x H x W

        作用：
            把梯度从 pooling 输出传回 pooling 输入。
            每个 pooling window 的梯度只传给 forward 时最大值的位置。
        """
        # ----输入检测----
        if not isinstance(d_pool_output, FeatureMap):
            raise TypeError("max_pooling_backward 的输入必须是 FeatureMap")

        if self.out_pooling is None or self.max_index_mask is None:
            raise ValueError("必须先执行 max_pooling，才能做 max_pooling_backward")

        if d_pool_output.feature.shape != self.out_pooling.feature.shape:
            raise ValueError(
                f"梯度 shape 错误，期望 {self.out_pooling.feature.shape}，但得到 {d_pool_output.feature.shape}"
            )

        # ----赋值----
        input_feature = self.input.feature
        d_output = d_pool_output.feature

        channel = input_feature.shape[0]
        height = input_feature.shape[1]
        width = input_feature.shape[2]

        pool_size = self.pool_size
        stride = self.stride

        output_height = d_output.shape[1]
        output_width = d_output.shape[2]

        # 创建传回输入层的梯度
        d_input = np.zeros_like(input_feature, dtype=np.float32)

        # 利用mask矩阵进行还原
        for c in range(channel):
            for out_row in range(output_height):
                for out_col in range(output_width):
                    # 定位初始位置
                    start_row = out_row * stride
                    start_col = out_col * stride
                    # 取出当前 pooling window 对应的 mask
                    region_mask = self.max_index_mask[
                        c,
                        start_row:start_row + pool_size,
                        start_col:start_col + pool_size
                    ]
                    # 找到输出位置传回来的梯度
                    grad_value = d_output[c, out_row, out_col]
                    # 梯度给到有最大值位置
                    d_input[
                        c,
                        start_row:start_row + pool_size,
                        start_col:start_col + pool_size
                    ] += region_mask * grad_value

        # 返回回传的FeatureMap
        return FeatureMap(d_input, self.input.label)
