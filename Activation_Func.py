import numpy as np
from Data_Sturcture import FeatureMap, Vector


class ReLU:
    """
    Relu函数，可用来反向传播
    """
    def __init__(self, input_data):
        if not isinstance(input_data, (FeatureMap, Vector)):
            raise TypeError("ReLULayer 的输入必须是 FeatureMap 或 Vector")

        self.input = input_data
        self.output = None

    def forward(self):
        """
            y = max(0, x)
        """

        if isinstance(self.input, FeatureMap):
            relu_result = np.maximum(0, self.input.feature)
            self.output = FeatureMap(relu_result, self.input.label)
            return self.output

        elif isinstance(self.input, Vector):
            relu_result = np.maximum(0, self.input.vector)
            self.output = Vector(relu_result, self.input.label)
            return self.output

    def backward(self, d_output):
        """
        input > 0，梯度不变
        input <= 0，梯度变成 0
        """

        # FeatureMap
        if isinstance(self.input, FeatureMap):
            #---输入检查/赋值----
            if isinstance(d_output, FeatureMap):
                grad = d_output.feature
            else:
                grad = np.array(d_output, dtype=np.float32)
            if grad.shape != self.input.feature.shape:
                raise ValueError(
                    f"ReLU backward 梯度形状错误，期望 {self.input.feature.shape}，但得到 {grad.shape}"
                )
            #返回梯度
            d_input = grad * (self.input.feature > 0)

            return FeatureMap(d_input, self.input.label)

        # Vector
        elif isinstance(self.input, Vector):
            if isinstance(d_output, Vector):
                grad = d_output.vector
            else:
                grad = np.array(d_output, dtype=np.float32).reshape(-1)

            if grad.shape != self.input.vector.shape:
                raise ValueError(
                    f"ReLU backward 梯度形状错误，期望 {self.input.vector.shape}，但得到 {grad.shape}"
                )

            d_input = grad * (self.input.vector > 0)

            return Vector(d_input, self.input.label)


class SoftmaxCrossEntropyLoss:
    """
    Softmax + Cross Entropy Loss。

    softmax: p_i = exp(z_i) / sum(exp(z_j))
    CEL: loss = -np.log(probabilities[true_label] + 1e-12)

    输入：
        FC2 的 Vector (Logits)

    输出：
        loss: float
        d_logits: Vector
    """

    def __init__(self, logits: Vector, true_label=None):
        #---输入检查/赋值----
        if not isinstance(logits, Vector):
            raise TypeError("logits 必须是 Vector 类型")
        self.logits = logits

        if true_label is None:
            true_label = logits.label

        if true_label is None:
            raise ValueError("true_label 不能是 None")

        if not isinstance(true_label, int):
            raise TypeError("true_label 必须是整数")

        if true_label < 0 or true_label >= logits.length:
            raise ValueError(
                f"true_label 超出范围，label={true_label}, logits length={logits.length}"
            )

        self.true_label = true_label
        self.probabilities = None
        self.loss = None

    def forward(self):
        """
        计算 softmax 概率和 cross entropy loss
        """

        logits = self.logits.vector

        # 防止logits过大，减某个值，确保 exp 不溢出
        shifted_logits = logits - np.max(logits)
        # 对应公式p_i = exp(z_i) / sum(exp(z_j))
        exp_values = np.exp(shifted_logits)
        self.probabilities = exp_values / np.sum(exp_values)

        # 防止 log(0)
        eps = 1e-12
        # 对应公式 loss = -np.log(probabilities[true_label] + 1e-12)
        self.loss = -np.log(self.probabilities[self.true_label] + eps)
        # 返回值
        return self.loss

    def backward(self) -> Vector:
        """
        d_logits = probabilities - one_hot_label
        返回：Vector
        """
        # 防止概率空缺
        if self.probabilities is None:
            self.forward()
        # 预测概率 - 真实概率
        d_logits = self.probabilities.copy()
        d_logits[self.true_label] -= 1.0

        return Vector(d_logits, self.true_label)


def predict(logits: Vector) -> int:
    """
    根据 logits 预测类别。
    """

    if not isinstance(logits, Vector):
        raise TypeError("logits 必须是 Vector 类型")

    return int(np.argmax(logits.vector))
