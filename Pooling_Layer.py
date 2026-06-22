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

    def max_pooling(self) -> FeatureMap:
        """
        执行池化操作。

        输入：
            self.input.image: 1 x H x W
        输出：
            FeatureMap: 1 x [floor((H - pooling_s) / stride) + 1]
            x [floor((W - pooling_s) / stride) + 1]
        """
        # ----赋值----
        input_feature = self.input.feature
        channel = input_feature.shape[0]
        height = input_feature.shape[1]
        width = input_feature.shape[2]

        pool_size = self.pool_size
        stride = self.stride

        # ----计算输出尺寸----
        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("池化输出尺寸错误，请检查 pool_size 和 stride 是否合理")

        # 创建输出的 FeatureMap 矩阵
        output = np.zeros(
            (
                channel,
                output_height,
                output_width
            ),
            dtype=np.float32
        )

        # 对每一个 channel 单独做 pooling (当前是二维不执行)
        for c in range(channel):

            for out_row in range(output_height):
                for out_col in range(output_width):

                    # 当前 pooling 窗口左上角位置
                    start_row = out_row * stride
                    start_col = out_col * stride
                    # 对应目标区域
                    region = input_feature[c,
                        start_row:start_row + pool_size,
                        start_col:start_col + pool_size]
                    # 取区域最大值
                    output[c, out_row, out_col] = np.max(region)

        # 存成 FeatureMap类并返回
        self.out_pooling = FeatureMap(output, self.input.label)
        return self.out_pooling

    def flatten(self) -> Vector:
        """
        把 FeatureMap 拉平成 Vector。

        输入：
            FeatureMap.feature: C x H x W

        输出：
            Vector.vector: C * H * W
        """

        # 拉平feature
        feature = self.out_pooling.feature
        flatten_vector = feature.reshape(-1)

        # 保存，返回
        self.out_flatten = Vector(flatten_vector, self.input.label)
        return self.out_flatten
