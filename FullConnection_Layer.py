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
        self.output = None

        self.input_size = input_vector.length
        self.output_size = output_size

        # 初始化参数
        # He initialization，适合后面接 ReLU
        self.weights = (
                np.random.randn(self.output_size, self.input_size).astype(np.float32)
                * np.sqrt(2.0 / self.input_size)
        )
        # 初始偏置为0
        self.bias = np.zeros(self.output_size, dtype=np.float32)
        # 参数导数
        self.d_weights = None
        self.d_bias = None
        self.d_input = None


    def forward(self) -> Vector:
        """
        前向传播, 输出: self.output:Vector
        """
        x = self.input.vector

        # 公式 W · x + b
        output_vector = self.weights @ x + self.bias

        self.output = Vector(output_vector, self.input.label)
        return self.output


    def backward(self, d_output, learning_rate):
        """
        反向传播。
        forward: output = W · x + b
        backward:
            d_output = dL/dz
            dW = dL/dW
            db = dL/db
            d_input = dL/dx

        d_output: 来自后一层的偏导。
        learning_rate: 学习率。
        d_input: 继续回传的梯度偏导
        """

        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")

        # ----取出梯度----
        # grad_output = dL/dz
        if isinstance(d_output, Vector):
            grad_output = d_output.vector
        else:
            grad_output = np.array(d_output, dtype=np.float32).reshape(-1)

        if grad_output.shape[0] != self.output_size:
            raise ValueError(f"d_output 维度错误，期望 {self.output_size}，但得到 {grad_output.shape[0]}")

        # ----取出 forward 时的输入----
        x = self.input.vector

        # ----计算回传梯度----
        # dL/dW shape: output_size x input_size
        d_weights = grad_output.reshape(self.output_size, 1) @ x.reshape(1, self.input_size)
        # dL/db shape: output_size
        d_bias = grad_output.copy()
        # dL/dx shape: input_size
        # 计算回传的input
        # dL/dx = W^T · dL/dz
        d_input = self.weights.T @ grad_output

        # ----保存梯度----
        self.d_weights = d_weights
        self.d_bias = d_bias
        self.d_input = d_input

        # ----SGD(梯度下降)更新参数----
        # W = W - learning_rate × dW
        # b = b - learning_rate × db
        self.weights = self.weights - learning_rate * d_weights
        self.bias = self.bias - learning_rate * d_bias
        # 返回dL/dx 和 label
        return Vector(d_input, self.input.label)


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
