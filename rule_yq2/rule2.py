import pandas as pd
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def Not_Normalized_Data(train_loader, threshold_mean=0.2, threshold_samples=500, channel=1):
    means = []
    mean = 0
    image_cnt = 0
    cnt = 0
    for batch_idx, data in enumerate(train_loader):
        train_features, train_labels = data
        mean += torch.mean(train_features).item()
        # print(mean / (batch_idx + 1))
        size = list(train_labels.size())
        image_cnt += size[0]
        cnt += 1
        means.append(mean / (batch_idx + 1))
        if abs(mean / (batch_idx + 1)) >= threshold_mean and image_cnt >= threshold_samples:
            quants = [i for i in range(cnt)]
            dict = {'quants': quants, 'means': means}
            df = pd.DataFrame(dict)
            df.to_csv('./data2.csv', index=False)  # path需要修改
            # draw_point(quants, means)
            return False
    if image_cnt <= threshold_samples:
        print("The number of samples is not enough!")
        return False

    quants = [i for i in range(cnt)]
    dict = {'quants': quants, 'means': means}
    df = pd.DataFrame(dict)
    df.to_csv('./data2.csv', index=False)  # path需要修改
    # draw_point(quants, means)
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
    print(Not_Normalized_Data(train_loader, 0.001))
