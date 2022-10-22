import pandas as pd
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
import torch.nn.functional as F
import torch.nn as nn


def get_data(trial, tname, steps_range, modes):
    tensor = trial.tensor(tname)
    vals = []
    for s in steps_range:
        val = tensor.value(step_num=s, mode=modes).item()
        vals.append(val)
    return vals


def compare(pre_loss, loss, different_percent):
    diff = abs(pre_loss - loss) / pre_loss * 100
    print(diff)
    return True if diff >= different_percent else False


def loss_base_test(loss, steps, different, threshold, min_step):
    if not loss:
        return False
    pre_loss = loss[0]
    cnt = 0
    for i in range(1, len(steps)):
        if not compare(pre_loss, loss[i], different):
            cnt += 1
        else:
            cnt = 0
        if cnt >= min_step and loss[i] > threshold:
            return False
        pre_loss = loss[i]

    return True


def accuracy_test(accuracy_path, accuracy_threshold):
    accuracy = np.load(accuracy_path)
    train_accuracy = accuracy[0]
    test_accuracy = accuracy[1]
    if train_accuracy < accuracy_threshold or test_accuracy < accuracy_threshold:
        return False
    else:
        return True


def loss_test(trial_path, loss_threshold, min_steps, different_percent):
    trial = smd.create_trial(path=trial_path)
    loss_name_test = trial.tensor_names(collection='losses', mode=smd.modes.EVAL)
    steps_test = trial.steps(mode=smd.modes.EVAL)
    loss_test = get_data(trial, loss_name_test[0], steps_test, smd.modes.EVAL)

    loss_name_train = trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
    steps_train = trial.steps(mode=smd.modes.TRAIN)
    loss_train = get_data(trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

    dict1 = {'steps_test': steps_test, 'loss_test': loss_test}
    df = pd.DataFrame(dict1)
    df.to_csv('../data61.csv', index=False)

    dict2 = {'steps_train': steps_train, 'loss_train': loss_train}
    df = pd.DataFrame(dict2)
    df.to_csv('../data62.csv', index=False)

    if loss_base_test(loss_train, steps_train, different_percent, loss_threshold, min_steps) and loss_base_test(
            loss_test, steps_test, different_percent, loss_threshold, min_steps):
        return True
    else:
        return False


def Underfitting(method_choose, accuracy_path, accuracy_threshold, trial_path, loss_threshold=0.1, min_steps=10,
                 different=0.01):
    if method_choose:
        return accuracy_test(accuracy_path, accuracy_threshold)
    else:
        return loss_test(trial_path, loss_threshold, min_steps, different)


if __name__ == "__main__":
    accuracy_path = '../accuracy.npy'
    path = "../data/tmp/testing/demo"
    print(Underfitting(0, accuracy_path, 90, path))
