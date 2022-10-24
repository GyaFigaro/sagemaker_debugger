import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils
import torch
import smdebug as smd

class Rule_Datasets():
    def __init__(self):
        super().__init__()

    def input_balance(self, train_loader, threshold_imbalance=10):
        labels, count = data_process(train_loader)
        max_class_num = max(count)
        min_class_num = min(count)
        rate = max_class_num / min_class_num
        dict = {'labels': labels, 'count': count}
        df = pd.DataFrame(dict)
        df.to_csv('./debug_info/data.csv', index=False)  # path需要修改
        if rate >= threshold_imbalance:
            f = open('./debug_info/result1.txt', 'w')
            f.write("False")
            f.close()
            return False
        else:
            f = open('./debug_info/result1.txt', 'w')
            f.write("True")
            f.close()
            return True

    def Not_Normalized_Data(self, train_loader, threshold_mean=0.2, threshold_samples=500, channel=1):
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
                df.to_csv('./debug_info/data2.csv', index=False)  # path需要修改
                f = open('./debug_info/result2.txt', 'w')
                f.write("1")
                f.close()
                return False
        if image_cnt <= threshold_samples:
            f = open('./debug_info/result2.txt', 'w')
            f.write("2")
            f.close()
            return False

        quants = [i for i in range(cnt)]
        dict = {'quants': quants, 'means': means}
        df = pd.DataFrame(dict)
        df.to_csv('./debug_info/data2.csv', index=False)  # path需要修改
        f = open('./debug_info/result2.txt', 'w')
        f.write("0")
        f.close()
        return True

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


