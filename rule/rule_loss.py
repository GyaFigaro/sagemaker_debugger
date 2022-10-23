import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils
import smdebug as smd
import json

class Rule_Loss():
    def __init__(self, base_trial):
        super().__init__()
        self.base_trial = base_trial
        self.epoch_info = load_epochfile()

    # different_percent: float 0.0~100 percents
    # increase_threhold_percent: float 0~100 percents
    # num_steps: int 0~
    def Loss_Not_Decreasing(self, tensor_regex=None, use_losses_collection=True, num_steps=10, different_percent=0.1,
                            increase_threshold_percent=5, patience=5):
        # input check
        # data process
        loss_name = self.base_trial.tensor_names(collection="losses", mode=smd.modes.TRAIN)
        steps = self.base_trial.steps(mode=smd.modes.TRAIN)
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
                if not compare1(pre_loss, loss, different_percent, increase_threshold_percent):
                    count += 1
                else:
                    count = 0
                if count >= patience:
                    dict = {'steps': steps[:step_index+1], 'losses': losses[:step_index+1]}
                    df = pd.DataFrame(dict)
                    df.to_csv('./data3.csv', index=False)
                    # plot_loss(steps[:step_index+1],losses[:step_index+1])
                    self.epoch_info['loss_not_decrease'] = True
                    update_epochfile(self.epoch_info)
                    return False
                pre_loss = loss
                start += num_steps

        # data save
        # output
        dict = {'steps': steps, 'losses': losses}
        df = pd.DataFrame(dict)
        df.to_csv('./debug_info/data3.csv', index=False)
        # plot_loss(steps, losses)
        return True

    def Overfitting(self, start_step=0, patience=1, ratio_threshold=0.1):
        loss_name_test = self.base_trial.tensor_names(collection='losses', mode=smd.modes.EVAL)
        steps_test =self.base_trial.steps(mode=smd.modes.EVAL)
        loss_test = get_data(self.base_trial, loss_name_test[0], steps_test, smd.modes.EVAL)

        loss_name_train = self.base_trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
        steps_train = self.base_trial.steps(mode=smd.modes.TRAIN)
        loss_train = get_data(self.base_trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

        n = len(steps_test)
        m = len(steps_train)
        if start_step + n > m:
            print("start_step is out of range!")
            return False
        cnt = 0
        dict = {'steps': steps_test, 'test_losses': loss_test, 'train_losses':loss_train[start_step:start_step + n]}
        df = pd.DataFrame(dict)
        df.to_csv('./debug_info/data4.csv', index=False)
        # plot_loss2(loss_train[start_step:start_step + n + 1], loss_test, steps_train[start_step:start_step + n + 1],
                # steps_test)
        for i in range(n):
            ratio = abs(loss_train[i + start_step] - loss_test[i]) / loss_test[i]
            if ratio > ratio_threshold:
                cnt += 1
            if cnt > patience:
                self.epoch_info['overfitting'] = 1
                update_epochfile(self.epoch_info)
                return False
        return True

    def Underfitting(self, method_choose, accuracy_path, accuracy_threshold, loss_threshold=0.1, min_steps=10,
                    different=0.01):
        if method_choose:
            return accuracy_test(accuracy_path, accuracy_threshold, self.epoch_info)
        else:
            return loss_test(self.base_trial, loss_threshold, min_steps, different, self.epoch_info)

    def Classifier_Confusion(self, category_no, labels, predictions, min_diag=0.9, max_off_diag=0.1):
        cnt = count(labels, predictions, category_no)
        #print(cnt)
        result = calculate(cnt,category_no)
        df = pd.DataFrame(result)
        df.to_csv('./debug_info/data5.csv', index=category_no, header=category_no)  # path需要修改
        #draw_table(result, category_no)
        for i in range(category_no):
            if result[i][i] < min_diag:
                return False
            for j in range(category_no):
                if result[j][i] > max_off_diag:
                    return False
        return True


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


def compare1(pre_loss, loss, different_percent, increase_threshold_percent):
    if pre_loss > loss:
        diff = (pre_loss - loss) / pre_loss * 100
        return True if diff >= different_percent else False
    else:
        diff = (loss - pre_loss) / pre_loss * 100
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

def compare2(pre_loss, loss, different_percent):
    diff = abs(pre_loss - loss) / pre_loss * 100
    return True if diff >= different_percent else False

def loss_base_test(loss, steps, different, threshold, min_step):
    if not loss:
        return False
    pre_loss = loss[0]
    cnt = 0
    for i in range(1, len(steps)):
        if not compare2(pre_loss, loss[i], different):
            cnt += 1
        else:
            cnt = 0
        if cnt >= min_step and loss[i] > threshold:
            return False
        pre_loss = loss[i]
    return True

def accuracy_test(accuracy_path, accuracy_threshold, epoch_info):
    accuracy = np.load(accuracy_path)
    train_accuracy = accuracy[0]
    test_accuracy = accuracy[1]
    if train_accuracy < accuracy_threshold or test_accuracy < accuracy_threshold:
        epoch_info['underfitting'] = True
        update_epochfile(epoch_info)
        return False
    else:
        return True


def loss_test(trial, loss_threshold, min_steps, different_percent, epoch_info):
    loss_name_test = trial.tensor_names(collection='losses', mode=smd.modes.EVAL)
    steps_test = trial.steps(mode=smd.modes.EVAL)
    loss_test = get_data(trial, loss_name_test[0], steps_test, smd.modes.EVAL)

    loss_name_train = trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
    steps_train = trial.steps(mode=smd.modes.TRAIN)
    loss_train = get_data(trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

    dict1 = {'steps_test': steps_test, 'loss_test': loss_test}
    df = pd.DataFrame(dict1)
    df.to_csv('./debug_info/data61.csv', index=False)

    dict2 = {'steps_train': steps_train, 'loss_train': loss_train}
    df = pd.DataFrame(dict2)
    df.to_csv('./debug_info/data62.csv', index=False)

    if loss_base_test(loss_train, steps_train, different_percent, loss_threshold, min_steps) and loss_base_test(
            loss_test, steps_test, different_percent, loss_threshold, min_steps):
        epoch_info['underfitting'] = True
        update_epochfile(epoch_info)
        return True
    else:
        return False

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

def load_epochfile():
    with open("./debug_info/epoch_info.json",'r') as load_f:
        epoch_info = json.load(load_f)
    return epoch_info

def update_epochfile(epoch_info):
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epoch_info, f)
        
