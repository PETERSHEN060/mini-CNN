# 手写 CNN 数字识别项目

本项目使用 Python 和 NumPy 从零实现了一个简单的卷积神经网络，用于识别 MNIST 手写数字图片。项目没有使用 PyTorch、TensorFlow 等深度学习框架，主要目的是理解 CNN 的前向传播、反向传播和参数更新过程。

## 项目结构

* `Data_Sturcture.py`：定义图片矩阵、特征图、向量和卷积核等基础数据结构
* `Convolution_Layer.py`：实现卷积层的前向传播和反向传播
* `Pooling_Layer.py`：实现最大池化、Flatten 以及对应的反向传播
* `FullConnection_Layer.py`：实现全连接层的前向传播、反向传播和参数更新
* `Activation_Func.py`：实现 ReLU、Softmax 和 Cross Entropy Loss
* `Read_Data_set.py`：读取 MNIST 数据集
* `CNNNetwork.py`：封装完整 CNN 网络
* `train.py`：训练模型
* `main.py`读取图片进行预测
* 'data_set' 文件夹放置数据集
* `model`用于存放已训练的模型参数

## 网络结构

```text
Input 1×28×28
→ Conv 3×3
→ ReLU
→ MaxPool 2×2
→ Flatten
→ Fully Connected 196→64
→ ReLU
→ Fully Connected 64→10
→ Softmax + CrossEntropy
```

## 功能

* 读取 MNIST 数据集
* 手写 CNN 前向传播
* 手写反向传播与 SGD 参数更新
* 保存训练好的权重为 `.npz` 文件
* 读取训练好的权重进行预测
* 支持读取本地 28×28 图片进行数字识别

## 运行方式

训练模型：

```bash
python train.py
```

预测自定义图片前，请先确认已经保存好模型权重，例如：

```text
cnn_weights.npz
```

然后在主程序中设置图片路径并运行预测函数。

## 说明

本项目主要用于学习 CNN 的底层原理，因此代码更注重可读性和手写实现过程，而不是训练速度和最终准确率。









!!! 估计是因为参数量太少了，效果依托，梦到那个猜那个
