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

        # ----反向传播----
        self.d_kernel = None
        self.d_padded_input = None
        self.d_input = None


    def kernel_padding(self) -> DataMatrix:
        """
        进行padding操作

        对 input matrix 进行 padding，然后赋值给 self.padded_input。
        返回 padding 后的 DataMatrix
        """
        # ----赋值----
        input_image = self.input.image
        channel = input_image.shape[0]
        height = input_image.shape[1]
        width = input_image.shape[2]

        padding = self.padding

        # ---- padding = 0，直接返回原图----
        if padding == 0:
            self.padded_input = Data_Sturcture.DataMatrix(
                input_image.copy(),self.input.label)
            return self.padded_input

        # 创建一个全 0 的新矩阵
        padded_image = np.zeros(
            (channel, height + 2 * padding,
                width + 2 * padding), dtype=np.float32)
        # 把原图放到中间
        padded_image[ : , padding:padding + height,
            padding:padding + width] = input_image

        # 存成 DataMatrix
        self.padded_input = Data_Sturcture.DataMatrix(padded_image, self.input.label)

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
        self.convolution_output = FeatureMap(output_matrix, self.input.label)
        return self.convolution_output


    def backward(self, d_output: FeatureMap, learning_rate):
        """
        反向传播。
        backward:
            d_output = dL/dY
            d_kernel = dL/dK
            d_padded_input = dL/dx

        d_output: 来自后一层的梯度，
        learning_rate: 学习率。
        d_input: 继续回传的梯度偏导
        更新：self.kernel.weights
        """
        # ----输入检测----
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")

        if not isinstance(d_output, FeatureMap):
            raise TypeError("d_output 必须是 FeatureMap 类型")

        if self.padded_input is None:
            raise ValueError("必须先执行 forward convolution，才能执行 backward")

        if self.convolution_output is None:
            raise ValueError("必须先执行 convolution，才能执行 backward")

        # 取出卷积前的image X
        padded_image = self.padded_input.image
        # 取出卷积核和回传梯度
        kernel_weights = self.kernel.weights
        # dL/dy
        d_y = d_output.feature
        # ----检查 shape----
        if d_y.shape != self.convolution_output.feature.shape:
            raise ValueError(
                f"d_output shape 错误，期望 {self.convolution_output.feature.shape}，但得到 {d_y.shape}")

        channel = padded_image.shape[0]
        padded_height = padded_image.shape[1]
        padded_width = padded_image.shape[2]

        kernel_size = self.kernel.kernel_size
        stride = self.stride
        padding = self.padding

        if channel != 1:
            raise ValueError("当前版本只支持单通道卷积 backward")

        # 初始化dK, dx
        # dL/dK
        d_kernel = np.zeros_like(kernel_weights, dtype=np.float32)
        # dL/dx
        d_padded_input = np.zeros_like(padded_image, dtype=np.float32)
        d_y_height = d_y.shape[1]
        d_y_width = d_y.shape[2]

        for d_y_row in range(d_y_height):
            for d_y_col in range(d_y_width):

                # 当前窗口左上角在 dy 中的位置
                start_row = d_y_row * stride
                start_col = d_y_col * stride
                # 取出卷积前的对应image
                region_x = padded_image[
                    0,
                    start_row:start_row + kernel_size,
                    start_col:start_col + kernel_size
                ]
                # 取出区域回传的梯度值
                d_y_value = d_y[0, d_y_row, d_y_col]
                # 计算dL/dk = dL/dy · X
                d_kernel += d_y_value * region_x
                # 计算dL/dx = dL/dy · K
                d_padded_input[
                    0,
                    start_row:start_row + kernel_size,
                    start_col:start_col + kernel_size
                ] += d_y_value * kernel_weights

        # ----去掉 padding，得到和原始 input 一样大小的梯度----
        if padding == 0:
            d_input = d_padded_input
        else:
            d_input = d_padded_input[
                :,
                padding:padded_height - padding,
                padding:padded_width - padding
            ]

        # ----记录梯度----
        self.d_kernel = d_kernel
        self.d_padded_input = d_padded_input
        self.d_input = d_input

        # ----更新 kernel 参数----
        self.kernel.weights = self.kernel.weights - learning_rate * d_kernel

        # 返回 FeatureMap
        return FeatureMap(d_input, self.input.label)
