import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
import torch.nn.functional as F
import torch.nn as nn


def plot_loss(x, y):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, y, '.-')
    plt_title = 'train_loss'
    plt.title(plt_title)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


def get_data(trial, tname, steps_range):
    tensor = trial.tensor(tname)
    vals = []
    for s in steps_range:
        val = tensor.value(step_num=s).item()
        vals.append(val)
    return vals


def compare(pre_loss, loss, different_percent, increase_threshold_percent):
    if pre_loss > loss:
        diff = (pre_loss - loss) / pre_loss * 100
        print(diff)
        return True if diff >= different_percent else False
    else:
        diff = (loss - pre_loss) / pre_loss * 100
        print(diff)
        return True if diff <= increase_threshold_percent else False


# different_percent: float 0.0~100 percents
# increase_threhold_percent: float 0~100 percents
# num_steps: int 0~
def Loss_Not_Decreasing(trial_path, tensor_regex=None, use_losses_collection=True, num_steps=10, different_percent=0.1,
                        increase_threshold_percent=5, patience=5):
    # input check
    # data process
    trial = smd.create_trial(path=trial_path)
    loss_name = trial.tensor_names(collection="losses", mode=smd.modes.TRAIN)
    steps = trial.steps(mode=smd.modes.TRAIN)
    print(steps)
    if use_losses_collection == True and loss_name[0]:
        losses = get_data(trial, loss_name[0], steps)
    else:
        losses = []
        for tensor in tensor_regex:
            losses.append(tensor.item())
    # main module
    start = 0
    current_loss = 0
    step_index = 0
    pre_loss = losses[step_index]
    n = len(steps)
    start += num_steps
    count = 0
    while step_index < n:
        if steps[step_index] < start:
            step_index += 1
            continue
        else:
            loss = losses[step_index]
            start = steps[step_index]
            print(pre_loss, loss)
            if not compare(pre_loss, loss, different_percent, increase_threshold_percent):
                count += 1
            else:
                count = 0
            if count >= patience:
                plot_loss(steps[:step_index+1],losses[:step_index+1])
                return False
            pre_loss = loss
            start += num_steps

    # data save
    # output
    plot_loss(steps, losses)
    return True


if __name__ == "__main__":
    path = "./tmp_yq/testing/demo"
    print(Loss_Not_Decreasing(path, increase_threshold_percent=100))
