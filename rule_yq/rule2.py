import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def draw_point(labels, quants):
    zeros = [0] * len(labels)
    plt.figure(figsize=(200, 100), dpi=100)
    plt.plot(labels, quants, c='red', label="均值")
    plt.plot(labels, zeros, c='green', linestyle='--', label="0")
    plt.scatter(labels, quants, c='red')
    plt.yticks([i * 0.01 for i in range(-5, 5)])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("epoch", fontdict={'size': 16})
    plt.ylabel("Average", fontdict={'size': 16})
    plt.title("Average of samples", fontdict={'size': 20})
    plt.show()


def Not_Normalized_Data(train_loader, threshold_mean=0.2, threshold_samples=500, channel=1):
    means = []
    mean = 0
    image_cnt = 0
    cnt = 0
    for batch_idx, data in enumerate(train_loader):
        train_features, train_labels = data
        mean += torch.mean(train_features).data
        size = list(train_labels.size())
        image_cnt += size[0]
        cnt += 1
        means.append(mean / (batch_idx + 1))
        if abs(mean / (batch_idx + 1)) >= threshold_mean and image_cnt >= threshold_samples:
            quants = [i for i in range(cnt)]
            draw_point(quants, means)
            return False
    if image_cnt <= threshold_samples:
        print("The number of samples is not enough!")
        return False
    quants = [i for i in range(cnt)]
    draw_point(quants, means)
    return True


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=64,
        shuffle=True,  # 打乱数据
    )
    print(Not_Normalized_Data(train_loader))
    print(Not_Normalized_Data(train_loader,0.001))