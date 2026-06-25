import numpy as np

from Data_Sturcture import DataMatrix, Kernel, Vector
from Convolution_Layer import ConvolutionLayer
from Activation_Func import ReLU, SoftmaxCrossEntropyLoss, predict
from Pooling_Layer import MaxPoollingLayer
from FullConnection_Layer import FullyConnectedLayer



class CNNNetwork:
    """
    CNN 结构：

    Input: 1 x 28 x 28
    -> Conv 3x3, stride=1, padding=1
    -> ReLU
    -> MaxPool 2x2, stride=2
    -> Flatten
    -> FC1: 196 -> 64
    -> ReLU
    -> FC2: 64 -> 10
    -> logits
    """

    def __init__(self):
        # 可训练参数
        self.kernel = Kernel(kernel_size=3)

        # FC 层需要等第一次 forward 得到 flatten vector 后再初始化
        self.fc1 = None
        self.fc2 = None

        # 保存 forward 过程中的层，backward 要用
        self.conv_layer = None
        self.relu1_layer = None
        self.pool_layer = None
        self.relu2_layer = None

        # 保存输出
        self.logits = None
        self.loss_layer = None

    def forward(self, data: DataMatrix, training=True):
        """
        前向传播。

        data: DataMatrix，一张  MNIST 图片
        """

        # 1. Convolution
        self.conv_layer = ConvolutionLayer(
            data=data,
            kernel=self.kernel,
            stride=1
        )
        conv_output = self.conv_layer.convolution()

        # 2. convolution 之后 ReLU
        self.relu1_layer = ReLU(conv_output)
        relu1_output = self.relu1_layer.forward()

        # 3. Max Pooling
        self.pool_layer = MaxPoollingLayer(
            relu1_output,
            pool_size=2,
            stride=2
        )
        self.pool_layer.max_pooling_forward()

        # 4. Flatten
        flatten_output = self.pool_layer.flatten_forward()

        # 5. FC1: 196 -> 64
        if self.fc1 is None:
            self.fc1 = FullyConnectedLayer(
                input_vector=flatten_output,
                output_size=64
            )
        else:
            self.fc1.set_input(flatten_output)

        fc1_output = self.fc1.forward()

        # 6. ReLU after FC1
        self.relu2_layer = ReLU(fc1_output)
        relu2_output = self.relu2_layer.forward()

        # 7. FC2: 64 -> 10
        if self.fc2 is None:
            self.fc2 = FullyConnectedLayer(
                input_vector=relu2_output,
                output_size=10
            )
        else:
            self.fc2.set_input(relu2_output)

        self.logits = self.fc2.forward()

        return self.logits

    def compute_loss(self, true_label=None):
        """
        输入 Logits 计算 softmax cross entropy loss。
        """

        if self.logits is None:
            raise ValueError("必须先执行 forward，才能计算 loss")

        self.loss_layer = SoftmaxCrossEntropyLoss(
            logits=self.logits,
            true_label=true_label
        )

        loss = self.loss_layer.forward()
        return loss

    def backward(self, learning_rate):
        """
        反向传播。
            loss
            -> FC2
            -> ReLU2
            -> FC1
            -> Flatten
            -> MaxPool
            -> ReLU1
            -> Conv
        """

        if self.loss_layer is None:
            raise ValueError("必须先执行 compute_loss，才能 backward")

        # 1. loss -> logits
        d_logits = self.loss_layer.backward()

        # 2. FC2 backward: 10 -> 64
        d_fc2_input = self.fc2.backward(
            d_logits,
            learning_rate=learning_rate
        )

        # 3. ReLU2 backward: 64 -> 64
        d_relu2_input = self.relu2_layer.backward(d_fc2_input)

        # 4. FC1 backward: 64 -> 196
        d_fc1_input = self.fc1.backward(
            d_relu2_input,
            learning_rate=learning_rate
        )

        # 5. Flatten backward: 196 -> 1 x 14 x 14
        d_flatten_input = self.pool_layer.flatten_backward(d_fc1_input)

        # 6. MaxPool backward: 1 x 14 x 14 -> 1 x 28 x 28
        d_pool_input = self.pool_layer.max_pooling_backward(d_flatten_input)

        # 7. ReLU1 backward: 1 x 28 x 28 -> 1 x 28 x 28
        d_relu1_input = self.relu1_layer.backward(d_pool_input)

        # 8. Conv backward: update kernel
        d_conv_input = self.conv_layer.backward(
            d_relu1_input,
            learning_rate=learning_rate
        )

        return d_conv_input

    def predict_one(self, data):
        """
        预测一张图片。
        """
        logits = self.forward(data, training=False)
        return predict(logits)

    def train_one(self, data, learning_rate):
        """
        训练一张图片。
        返回：loss, prediction
        """

        logits = self.forward(data, training=True)

        loss = self.compute_loss(data.label)

        prediction = predict(logits)

        self.backward(learning_rate)

        return loss, prediction

    def evaluate(self, dataset):
        """
        在 dataset 上计算准确率。
        """

        correct = 0

        for data in dataset:
            prediction = self.predict_one(data)

            if prediction == data.label:
                correct += 1

        accuracy = correct / len(dataset)

        return accuracy

    def save_weights(self, file_path="model/cnn_weight.npz"):
        """
        保存训练好的模型参数：
            kernel.weights
            fc1.weights
            fc1.bias
            fc2.weights
            fc2.bias
        """

        if self.fc1 is None or self.fc2 is None:
            raise ValueError("FC 层还没有初始化，必须至少 forward/train 一次后才能保存权重")

        np.savez(
            file_path,
            kernel_weights=self.kernel.weights,
            fc1_weights=self.fc1.weights,
            fc1_bias=self.fc1.bias,
            fc2_weights=self.fc2.weights,
            fc2_bias=self.fc2.bias
        )

        print(f"模型权重已保存到: {file_path}")

    def load_weights(self, file_path="model/cnn_weight.npz"):
        """
        读取已经保存好的模型参数。
        """
        data = np.load(file_path)

        # 1. 读取卷积核
        kernel_weights = data["kernel_weights"].astype(np.float32)

        self.kernel = Kernel(kernel_size=kernel_weights.shape[0])
        self.kernel.weights = kernel_weights

        # 2. 读取 FC1
        fc1_weights = data["fc1_weights"].astype(np.float32)
        fc1_bias = data["fc1_bias"].astype(np.float32)

        fc1_input_size = fc1_weights.shape[1]
        fc1_output_size = fc1_weights.shape[0]

        dummy_fc1_input = Vector(
            np.zeros(fc1_input_size, dtype=np.float32)
        )

        self.fc1 = FullyConnectedLayer(
            input_vector=dummy_fc1_input,
            output_size=fc1_output_size
        )

        self.fc1.weights = fc1_weights
        self.fc1.bias = fc1_bias

        # 3. 读取 FC2
        fc2_weights = data["fc2_weights"].astype(np.float32)
        fc2_bias = data["fc2_bias"].astype(np.float32)

        fc2_input_size = fc2_weights.shape[1]
        fc2_output_size = fc2_weights.shape[0]

        dummy_fc2_input = Vector(
            np.zeros(fc2_input_size, dtype=np.float32)
        )

        self.fc2 = FullyConnectedLayer(
            input_vector=dummy_fc2_input,
            output_size=fc2_output_size
        )

        self.fc2.weights = fc2_weights
        self.fc2.bias = fc2_bias

        print(f"模型权重已从 {file_path} 读取完成")
