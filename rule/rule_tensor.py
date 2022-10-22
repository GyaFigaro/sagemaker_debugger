from email.mime import base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils
import smdebug as smd
import json

class Rule_Tensor():
    def __init__(self, base_trial, steps,
                 rtol=1e-05, atol=1e-08, 
                 threshold_layer=0.4):
        super().__init__()
        self.base_trial = base_trial
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.threshold_layer = float(threshold_layer)
        self.steps = steps
        self.path = './debug_info/tensor'
        self.epoch_info = load_epochfile()

    # 制作数据表格
    def make_data_chart(self, step, dicts, rule_path):
        chart_path = rule_path + '/' + str(step) + '.csv'
        df = pd.DataFrame.from_dict(data=dicts, orient='columns')
        df.to_csv(chart_path, header=True, sep=' ', index=False)
        # print(plot_path)
        # x_label = list(dicts.keys())
        # y_label = list(dicts.values())
        # plt.figure(figsize=(100,40), dpi=90)
        # plt.plot(x_label, y_label, lw=2, ls='-', c='r', alpha=0.1)
        # plt.savefig(str(plot_path), dpi=90, bbox_inches = 'tight')
        # plt.show()

    # 制作结果表格
    def make_result_chart(self, steps, dicts, rule_path):
        chart_path = rule_path + '/result.csv'
        dict_0 = {'step': steps}
        cnt = 1
        for key in dicts:
            dict_0[key] = dicts[key]
            cnt += 1
        df = pd.DataFrame(dict_0)
        df.to_csv(chart_path, header=True, sep=' ', index=False)
    
    # 计算张量中0值的百分比
    def compute_zero_values(self, tensor):
        t = tensor.reshape(-1)
        size_t = np.size(t)
        cnt = 0
        for i in range(size_t):
            if t[i].item() == 0:
                cnt += 1
        return float(cnt/size_t)

    # step中获取数据
    def invoke_at_step(self, cur_step, layer_names, rule_id, last_step=None):
        if rule_id == 12:
            step_zeros = list()
            for tname in layer_names:
                tensor = self.base_trial.tensor(tname).value(step_num=cur_step, mode=smd.modes.TRAIN)
                percent = self.compute_zero_values(tensor)
                if percent >= self.threshold_layer:
                    step_zeros.append([tname, percent, True])
                else :
                    step_zeros.append([tname, percent, False])
            return step_zeros

        if rule_id == 13:
            step_tensors = list()
            for tname in layer_names:
                last_t = self.base_trial.tensor(tname).value(step_num=last_step, mode=smd.modes.TRAIN)
                cur_t = self.base_trial.tensor(tname).value(step_num=cur_step, mode=smd.modes.TRAIN)
                is_unchanged = np.allclose(
                    last_t, cur_t, 
                    rtol=self.rtol, 
                    atol=self.atol,
                    equal_nan=False)
                step_tensor = (tname, is_unchanged)
                step_tensors.append(step_tensor)
            return step_tensors
    
    # AVZ接口
    def all_values_zero(self):
        path = self.path + '/AllZeroValues'
        if not os.path.exists(path):
            os.makedirs(path)
        # print("steps: ", self.steps)
        layer_names = self.base_trial.tensor_names(regex="output", mode=smd.modes.TRAIN)
        percents = {'layers':[], 'percents':[]}
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()
          
        for step in self.steps:
            percents['layers'].clear()
            percents['percents'].clear()
            step_zeros = self.invoke_at_step(cur_step=step, 
                                             layer_names=layer_names, rule_id=12) 
            for step_zero in step_zeros:
                percents['layers'].append(step_zero[0])
                percents['percents'].append(step_zero[1])
                if step_zero[2] == True and step != self.steps[0]:
                    # print(step_zero[0], ": All values zero")
                    self.epoch_info['all_values_zero'] = True
                    update_epochfile(self.epoch_info)
                    results[step_zero[0]].append("All values zero")
                else:
                    # print(step_zero[0], ": Not all values zero")
                    results[step_zero[0]].append("Not all values zero")
            
            self.make_data_chart(step, percents, path)

        self.make_result_chart(self.steps, results, path)

    # VU接口
    def values_unchanged(self):
        path = self.path + '/ValuesUnchanged'
        if not os.path.exists(path):
            os.makedirs(path)

        print("steps: ", self.steps)
        layer_names = self.base_trial.tensor_names(collection="weights", mode=smd.modes.TRAIN)
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()

        last_step = self.steps[0]
        for step in self.steps: 
            step_tensors = self.invoke_at_step(last_step=last_step, cur_step=step, 
                                              layer_names = layer_names, rule_id=13) 
            for step_var in step_tensors:
                if step_var[1] == True and step != self.steps[0]:
                    # print(step_var[0], ": Tensors were unchanged")
                    self.epoch_info['tensors_unchanged'] = True
                    update_epochfile(self.epoch_info)
                    results[step_var[0]].append("Tensors were unchanged")
                else :
                    # print(step_var[0], ": Tensors changed properly")
                    results[step_var[0]].append("Tensors changed properly")
            last_step = step

        self.make_result_chart(self.steps, results, path)


def load_epochfile():
    with open("./debug_info/epoch_info.json",'r') as load_f:
        epoch_info = json.load(load_f)
    return epoch_info

def update_epochfile(epoch_info):
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epoch_info, f)