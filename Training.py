import numpy as np

from Read_Data_set import read_mnist_dataset
from CNN_Networks import CNNNetwork


def train():
    train_images_path = "data_set/training_set/train-images-idx3-ubyte.gz"
    train_labels_path = "data_set/training_set/train-labels-idx1-ubyte.gz"

    test_images_path = "data_set/testing_set/t10k-images-idx3-ubyte.gz"
    test_labels_path = "data_set/testing_set/t10k-labels-idx1-ubyte.gz"

    # 每个epoch读取数量
    train_dataset = read_mnist_dataset(
        train_images_path,
        train_labels_path,
        limit=10000
    )

    test_dataset = read_mnist_dataset(
        test_images_path,
        test_labels_path,
        limit=2000
    )

    model = CNNNetwork()

    epochs = 5
    learning_rate = 0.001

    for epoch in range(epochs):
        # 每轮打乱训练集
        np.random.shuffle(train_dataset)

        total_loss = 0.0
        correct = 0

        for index, data in enumerate(train_dataset):
            loss, prediction = model.train_one(
                data,
                learning_rate=learning_rate
            )

            total_loss += loss

            if prediction == data.label:
                correct += 1

            # 每 50 张打印一次
            if (index + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Step {index + 1}/{len(train_dataset)}, "
                    f"Current loss: {loss:.4f}"
                )

        train_loss = total_loss / len(train_dataset)
        train_accuracy = correct / len(train_dataset)

        test_accuracy = model.evaluate(test_dataset)

        print("=" * 50)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("=" * 50)

    # 训练结束后保存
    model.save_weights("model/cnn_weight2.npz")

if __name__ == "__main__":
    train()
