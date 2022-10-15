import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils
import smdebug as smd

class Rule_Loss():
    def __init__(self, base_trial):
        super().__init__()
        self.base_trial = base_trial

    # different_percent: float 0.0~100 percents
    # increase_threhold_percent: float 0~100 percents
    # num_steps: int 0~
    def Loss_Not_Decreasing(self, tensor_regex=None, use_losses_collection=True, num_steps=10, different_percent=0.1,
                            increase_threshold_percent=5, patience=5):
        # input check
        # data process
        loss_name = self.base_trial.tensor_names(collection="losses", mode=smd.modes.TRAIN)
        steps = self.base_trial.steps(mode=smd.modes.TRAIN)
        print(steps)
        if use_losses_collection == True and loss_name[0]:
            losses = get_data(self.base_trial, loss_name[0], steps, smd.modes.TRAIN)
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
                    dict = {'steps': steps[:step_index+1], 'losses': losses[:step_index+1]}
                    df = pd.DataFrame(dict)
                    df.to_csv('../data3.csv', index=False)
                    plot_loss(steps[:step_index+1],losses[:step_index+1])
                    return False
                pre_loss = loss
                start += num_steps

        # data save
        # output
        dict = {'steps': steps, 'losses': losses}
        df = pd.DataFrame(dict)
        df.to_csv('../data3.csv', index=False)
        plot_loss(steps, losses)
        return True

    def Overfitting(self, start_step=0, patience=1, ratio_threshold=0.1):
        loss_name_test = self.base_trialtensor_names(collection='losses', mode=smd.modes.EVAL)
        steps_test =self.base_trial.steps(mode=smd.modes.EVAL)
        loss_test = get_data(self.base_trial, loss_name_test[0], steps_test, smd.modes.EVAL)

        loss_name_train = self.base_trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
        steps_train = self.base_trial.steps(mode=smd.modes.TRAIN)
        loss_train = get_data(self.base_trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

        n = len(steps_test)
        m = len(steps_train)
        if start_step+n>m:
            print("start_step is out of range!")
            return False
        cnt = 0
        dict = {'steps': steps_test, 'test_losses': loss_test, 'train_losses':loss_train[start_step:start_step + n]}
        df = pd.DataFrame(dict)
        df.to_csv('../data4.csv', index=False)
        plot_loss2(loss_train[start_step:start_step + n + 1], loss_test, steps_train[start_step:start_step + n + 1],
                steps_test)
        for i in range(n):
            ratio = abs(loss_train[i + start_step] - loss_test[i]) / loss_test[i]
            if ratio > ratio_threshold:
                cnt += 1
            if cnt > patience:
                return False
        return True

    def Underfitting(self, method_choose, accuracy_path, accuracy_threshold, loss_threshold=0.1, min_steps=10,
                    different=0.01):
        if method_choose:
            return accuracy_test(accuracy_path, accuracy_threshold)
        else:
            return loss_test(self.base_trial, loss_threshold, min_steps, different)



def plot_loss(x, y):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, y, '.-')
    plt_title = 'train_loss'
    plt.title(plt_title)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


def get_data(trial, tname, steps_range, modes):
    tensor = trial.tensor(tname)
    vals = []
    for s in steps_range:
        val = tensor.value(step_num=s, mode=modes).item()
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

def plot_loss2(x1, x2, y1, y2):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), dpi=100)
    axs[0].plot(y2, x2, c='red', label="train_loss")
    axs[0].scatter(y2, x2, c='red')
    axs[1].plot(y1, x1, c='green', linestyle='--', label="test_loss")
    axs[1].scatter(y1, x1, c='green')
    plt_title = 'LOSS'
    axs[0].set_title('TEST_LOSS')
    axs[1].set_title('TRAIN_LOSS')
    # plt.title(plt_title)
    for i in range(2):
        axs[i].set_xlabel('step')
        axs[i].set_ylabel('loss')
    # plt.savefig(file_name)
    plt.show()

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


def loss_test(trial, loss_threshold, min_steps, different_percent):
    loss_name_test = trial.tensor_names(collection='losses', mode=smd.modes.EVAL)
    steps_test = trial.steps(mode=smd.modes.EVAL)
    loss_test = get_data(trial, loss_name_test[0], steps_test, smd.modes.EVAL)

    loss_name_train = trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
    steps_train = trial.steps(mode=smd.modes.TRAIN)
    loss_train = get_data(trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

    if loss_base_test(loss_train, steps_train, different_percent, loss_threshold, min_steps) and loss_base_test(
            loss_test, steps_test, different_percent, loss_threshold, min_steps):
        return True
    else:
        return False

