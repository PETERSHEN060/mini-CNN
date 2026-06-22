import numpy as np

from Read_Data_set import read_mnist_dataset
from Data_Sturcture import Kernel
from Convolution_Layer import ConvolutionLayer
from Pooling_Layer import MaxPoollingLayer
from FullConnection_Layer import FullyConnectedLayer
from Activation_Func import Relu


def predict_one_image(data, kernel, fc1=None, fc2=None):
    """
    对一张 MNIST 图片执行一次完整 forward。

    网络结构：
        DataMatrix
        -> Conv
        -> ReLU
        -> MaxPool
        -> Flatten
        -> FC1
        -> ReLU
        -> FC2
        -> logits
        -> argmax

    参数：
        data: DataMatrix，一张图片
        kernel: Kernel，共用同一个卷积核
        fc1: 第一层全连接层，可复用
        fc2: 第二层全连接层，可复用

    返回：
        prediction: 预测类别
        logits: 最后一层输出 Vector
        fc1, fc2: 返回全连接层对象，方便后续图片复用同一套权重
    """

    # 1. Convolution
    conv_layer = ConvolutionLayer(data=data, kernel=kernel, stride=1)
    conv_output = conv_layer.convolution()

    # 2. ReLU after convolution
    relu_conv_output = Relu(conv_output)

    # 3. Max Pooling
    pool_layer = MaxPoollingLayer(relu_conv_output, pool_size=2, stride=2)
    pool_layer.max_pooling()

    # 4. Flatten
    flatten_output = pool_layer.flatten()

    # 5. FC1: 196 -> 64
    if fc1 is None:
        fc1 = FullyConnectedLayer(input_vector=flatten_output, output_size = 64)
    else:
        fc1.set_input(flatten_output)

    fc1_output = fc1.forward()

    # 6. ReLU after FC1
    relu_fc1_output = Relu(fc1_output)

    # 7. FC2: 64 -> 10
    if fc2 is None:
        fc2 = FullyConnectedLayer(input_vector=relu_fc1_output, output_size=10)
    else:
        fc2.set_input(relu_fc1_output)

    # 8. 得到logits
    logits = fc2.forward()

    # 8. Prediction
    prediction = int(np.argmax(logits.vector))

    return prediction, logits, fc1, fc2


def main():
    test_images_path = "data_set/testing_set/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data_set/testing_set/t10k-labels-idx1-ubyte.gz"

    # 只读取 test set 前 10 张
    test_dataset = read_mnist_dataset(
        test_images_path,
        test_labels_path,
        limit=10
    )

    print("成功读取测试图片数量:", len(test_dataset))

    # 创建一个共享卷积核
    # 当前结构是单 kernel，所以卷积输出还是 1 x 28 x 28
    kernel = Kernel(kernel_size=3)

    # fc1 / fc2 先设为 None
    # 第一张图片 forward 时初始化权重
    # 后面 9 张图片复用同一套 fc 权重
    fc1 = None
    fc2 = None

    correct = 0

    print("\n开始预测 MNIST test set 前 10 张：\n")

    for index, data in enumerate(test_dataset):
        prediction, logits, fc1, fc2 = predict_one_image(
            data=data,
            kernel=kernel,
            fc1=fc1,
            fc2=fc2
        )

        true_label = data.label

        if prediction == true_label:
            correct += 1

        print(f"Image {index}: prediction = {prediction}, true label = {true_label}")

    accuracy = correct / len(test_dataset)

    print("\n前 10 张测试结果：")
    print("Correct:", correct)
    print("Total:", len(test_dataset))
    print("Accuracy:", accuracy)

    print("\n注意：当前模型没有训练，权重是随机初始化的，所以预测结果大概率是随机的。")


if __name__ == "__main__":
    main()
