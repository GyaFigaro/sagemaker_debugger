import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    df.to_csv('./data.csv', index=False)  # path需要修改
    if rate >= threshold_imbalance:
        #draw_bar(labels, count)
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
