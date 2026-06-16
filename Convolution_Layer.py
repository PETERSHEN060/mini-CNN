import numpy as np

import Data_Sturcture
from Data_Sturcture import DataMatrix, FeatureMap, Kernel

class ConvolutionLayer:
    def __init__(self, data: DataMatrix, kernel: Kernel, stride=1):
        self.input = data
        self.kernel = kernel

        self.padded_input = None

        self.stride = stride
        # padding 公式
        self.padding = (kernel.kernel_size - 1) // 2

        self.convolution_output = None


    def kernel_padding(self) -> DataMatrix:
        """
        进行padding操作

        对 input matrix 进行 padding，然后赋值给 self.padded_input。
        返回 padding 后的 DataMatrix
        """
        input_image = self.input.image

        channel = input_image.shape[0]
        height = input_image.shape[1]
        width = input_image.shape[2]

        padding = self.padding

        # padding = 0，不补边，直接复制原图
        if padding == 0:
            self.padded_input = Data_Sturcture.DataMatrix(
                input_image.copy(),
                self.input.label
            )
            return self.padded_input

        # 创建一个全 0 的新矩阵
        padded_image = np.zeros(
            (
                channel,
                height + 2 * padding,
                width + 2 * padding
            ),
            dtype=np.float32
        )

        # 把原图放到中间
        padded_image[
            :,
            padding:padding + height,
            padding:padding + width
        ] = input_image

        # 存成 DataMatrix
        self.padded_input = Data_Sturcture.DataMatrix(
            padded_image,
            self.input.label
        )

        return self.padded_input


    def convolution(self) -> FeatureMap:
        """
        执行卷积操作。

        输入：
            self.input.image: 1 x H x W

        padding 后：
            self.padded_input.image: 1 x padded_H x padded_W

        输出：
            FeatureMap: 1 x output_H x output_W
        """
        if self.padded_input is None:
            self.kernel_padding()

        # ----导入数据----
        padded_image = self.padded_input.image
        channel = padded_image.shape[0]
        padded_height = padded_image.shape[1]
        padded_width = padded_image.shape[2]

        kernel_size = self.kernel.kernel_size
        kernel_weights = self.kernel.weights
        stride = self.stride

        # ----异常值处理----
        # 检查颜色
        if channel != 1:
            raise ValueError(f"input channel should == 1")
        # 检查 kernel
        if kernel_weights.shape != (kernel_size, kernel_size):
            raise ValueError(
                f"kernel 格式错误，应该是 {(kernel_size, kernel_size)}，但得到 {kernel_weights.shape}"
            )
        # 检查 stride
        if stride <= 0:
            raise ValueError("stride 必须大于 0")


        # ----计算输出的feature map 的大小----
        output_height = (padded_height - kernel_size) // stride + 1
        output_width = (padded_width - kernel_size) // stride + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("卷积输出尺寸错误，请检查 kernel_size、padding、stride 是否合理")


        # 创建输出矩阵，保持 1 x H x W 的三维结构
        output_matrix = np.zeros(
            (
                1,
                output_height,
                output_width
            ),
            dtype=np.float32
        )
        # kernel 在 padded_input 上滑动
        for out_row in range(output_height):
            for out_col in range(output_width):

                # 当前窗口左上角在 padded_image 中的位置
                start_row = out_row * stride
                start_col = out_col * stride
                # 取出和 kernel 的对应目标区域
                region = padded_image[0,
                    start_row:start_row + kernel_size,
                    start_col:start_col + kernel_size]
                # 目标区域和 kernel 做点乘
                conv_value = np.sum(region * kernel_weights)
                # 存入output_matrix 的对应位置
                output_matrix[0, out_row, out_col] = conv_value

        # 存成 FeatureMap类并返回
        self.convolution_output = FeatureMap(
            output_matrix, self.input.label)
        return self.convolution_output

