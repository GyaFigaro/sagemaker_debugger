import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
import torch.nn.functional as F
import torch.nn as nn

predictions = []
labels = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("relu0", nn.ReLU())
        self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
        self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
        self.add_module("relu1", nn.ReLU())
        self.add_module("max_pool2", nn.MaxPool2d(2, stride=2))
        self.add_module("fc1", nn.Linear(4 * 4 * 50, 500))
        self.add_module("relu2", nn.ReLU())
        self.add_module("fc2", nn.Linear(500, 10))

    def forward(self, x):
        x = self.relu0(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu1(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def draw_table(vals, num):
    col = [i for i in range(0, num)]
    row = [i for i in range(0, num)]
    plt.figure(figsize=(30, 8))
    tab = plt.table(cellText=vals,
                    colLabels=col,
                    rowLabels=row,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center',
                    colWidths=[0.06] * num)
    tab.scale(1, 2)
    plt.axis('off')
    plt.show()


# 统计
def count(labels, predictions, category):
    cnt = [[0 for i in range(category)] for i in range(category)]
    m = len(labels)
    for i in range(m):
        size = list(labels[i].size())
        for j in range(size[0]):
            x, y = labels[i][j].item(), predictions[i][j].item()
            cnt[x][y] += 1
    return cnt


# 计算
def calculate(count, category):
    result = [[0 for i in range(category)] for i in range(category)]
    sum_diag = 0
    sum_non_diag = [0 for i in range(category)]
    for i in range(category):
        sum_diag += count[i][i]
        for j in range(category):
            sum_non_diag[i] += count[j][i]
    for i in range(category):
        for j in range(category):
            if j == i:
                result[i][j] = round(count[i][j]/sum_diag,4)
            else:
                result[j][i] = round(count[j][i]/sum_non_diag[i],4)

    return result


def Classifier_Confusion(category_no, labels, predictions, min_diag=0.9, max_off_diag=0.1):
    cnt = count(labels,predictions,category_no)
    print(cnt)
    result = calculate(cnt,category_no)
    draw_table(result, category_no)
    for i in range(category_no):
        if result[i][i] < min_diag:
            return False
        for j in range(category_no):
            if result[j][i] > max_off_diag:
                return False
    return True


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.append(pred)
            labels.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


if __name__ == "__main__":
    device = torch.device("cpu")
    model = torch.load('C:/190110328/pytorch_debug_demo/model.pt')
    model = model.to(device)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=1000,
        shuffle=True,
    )
    test(model, device, test_loader)
    print(Classifier_Confusion(10,labels,predictions))
