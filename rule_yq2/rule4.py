import matplotlib.pyplot as plt
import pandas as pd
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig


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


def Overfitting(base_trial, start_step=0, patience=1, ratio_threshold=0.1):
    trial = smd.create_trial(path=base_trial)
    loss_name_test = trial.tensor_names(collection='losses', mode=smd.modes.EVAL)
    steps_test = trial.steps(mode=smd.modes.EVAL)
    loss_test = get_data(trial, loss_name_test[0], steps_test, smd.modes.EVAL)

    loss_name_train = trial.tensor_names(collection='losses', mode=smd.modes.TRAIN)
    steps_train = trial.steps(mode=smd.modes.TRAIN)
    loss_train = get_data(trial, loss_name_train[0], steps_train, smd.modes.TRAIN)

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


if __name__ == "__main__":
    path = "./tmp/testing/demo"
    print(Overfitting(path, 20))
