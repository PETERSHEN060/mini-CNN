import numpy as np
from Data_Sturcture import Vector


class FullyConnectedLayer:
    """

    输入：
        Vector.vector: shape = (input_size,)

    输出：
        Vector.vector: shape = (output_size,)

    公式：
        output = (weights · input) + bias
    """

    def __init__(self, input_vector: Vector, output_size: int):
        # ----输入检查----
        if not isinstance(input_vector, Vector):
            raise TypeError("input_vector 必须是 Vector 类型")

        if not isinstance(output_size, int):
            raise TypeError("output_size 必须是整数")

        if output_size <= 0:
            raise ValueError("output_size 必须大于 0")

        # ----赋值----
        self.input = input_vector

        self.input_size = input_vector.length
        self.output_size = output_size

        # He initialization，适合后面接 ReLU
        self.weights = (
                np.random.randn(self.output_size, self.input_size).astype(np.float32)
                * np.sqrt(2.0 / self.input_size)
        )

        # 初始偏置为0
        self.bias = np.zeros(self.output_size, dtype=np.float32)

        self.output = None

    def forward(self) -> Vector:
        """
        前向传播, 输出:self.output:Vector
        """
        x = self.input.vector
        
        # 公式 W · x + b
        output_vector = self.weights @ x + self.bias

        self.output = Vector(output_vector, self.input.label)
        return self.output

    def set_input(self, input_vector: Vector):

        if not isinstance(input_vector, Vector):
            raise TypeError("input_vector 必须是 Vector 类型")
        if input_vector.length != self.input_size:
            raise ValueError(
                f"输入维度错误，期望 {self.input_size}，但得到 {input_vector.length}"
            )

        self.input = input_vector

    def show_info(self):
        print("FullyConnectedLayer")
        print("Input size:", self.input_size)
        print("Output size:", self.output_size)
        print("Weights shape:", self.weights.shape)
        print("Bias shape:", self.bias.shape)

        if self.output is not None:
            print("Output vector shape:", self.output.vector.shape)
            print("Output label:", self.output.label)
