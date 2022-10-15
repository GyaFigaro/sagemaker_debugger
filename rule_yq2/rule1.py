import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def draw_bar(labels, quants):
    width = 0.4
    ind = np.linspace(0.5, 9.5, 10)
    # make a square figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # Bar Plot
    ax.bar(ind - width / 2, quants, width, color='green')
    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_xlabel('Class')
    ax.set_ylabel('Amount')
    # title
    ax.set_title('The Number of Every Class', bbox={'facecolor': '0.8', 'pad': 5})
    plt.grid(True)
    plt.show()
    plt.savefig("input_class.jpg")
    plt.close()


def data_process(train_loader):
    labels = []
    count = []

    for batch_idx, data in enumerate(train_loader):
        train_features, train_labels = data
        for label in train_labels:
            if label not in labels:
                labels.append(label.item())
                # print(label.item())
                count.append(0)
            count[labels.index(label.item())] += 1

    return labels, count


def input_balance(train_loader, threshold_imbalance=10):
    labels, count = data_process(train_loader)
    max_class_num = max(count)
    min_class_num = min(count)
    rate = max_class_num / min_class_num
    dict = {'labels': labels, 'count': count}
    df = pd.DataFrame(dict)
    df.to_csv('../data.csv', index=False)
    if rate >= threshold_imbalance:
        draw_bar(labels, count)
        return False
    else:
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
    print(input_balance(train_loader))
    print(input_balance(train_loader, 1))
