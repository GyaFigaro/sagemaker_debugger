from re import T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils
import smdebug as smd
import json

class Rule_ActivationFunctions():
    def __init__(self, base_trial, steps,
                 threshold_layer = 0.4, 
                 threshold_tanh_min = -9.4999,
                 threshold_tanh_max = 9.4999,
                 threshold_sigmoid_min = -23,
                 threshold_sigmoid_max = 16.99999):
        super().__init__()
        self.base_trial = base_trial
        self.steps = steps  
        self.threshold_layer = float(threshold_layer)
        self.threshold_tanh_min = float(threshold_tanh_min)
        self.threshold_tanh_max = float(threshold_tanh_max)
        self.threshold_sigmoid_min = float(threshold_sigmoid_min)
        self.threshold_sigmoid_max = float(threshold_sigmoid_max)
        self.path = './debug_info/activationfunction'
        self.epoch_info = load_epochfile()
        
        
    def get_output_layer_names(self, regex):
        layer_names = list()
        for tname in self.base_trial.tensor_names(regex=regex, mode=smd.modes.EVAL):
            if tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names

    def make_data_chart(self, step, dicts, rule_path):
        chart_path = rule_path + '/' + str(step) + '.csv'
        df = pd.DataFrame.from_dict(data=dicts, orient='columns')
        df.to_csv(chart_path, index=False)
        # plot_path = rule_path + '/' + str(step) + '.png'
        # print(plot_path)
        # x_label = list(dicts.keys())
        # y_label = list(dicts.values())
        # plt.figure(figsize=(100,40), dpi=90)
        # plt.plot(x_label, y_label, lw=2, ls='-', c='r', alpha=0.1)
        # plt.savefig(str(plot_path), dpi=90, bbox_inches = 'tight')
        # plt.show()

    def make_result_chart(self, steps, dicts, rule_path):
        chart_path = rule_path + '/result.csv'
        df = pd.DataFrame(dicts)
        df.insert(0, "step", steps)
        df.to_csv(chart_path, index=False)
    
    def compute_dying_relus(self, last_tensor, cur_tensor):
        last_t = last_tensor.reshape(-1)
        cur_t = cur_tensor.reshape(-1)
        size_t = np.size(last_t)
        cnt = 0
        for i in range(size_t):
            if last_t[i].item() == 0 and cur_t[i].item() == 0:
                cnt += 1
        return float(cnt/size_t)

    def compute_saturation(self, regex, tensor):
        t = tensor.reshape(-1)
        size_t = np.size(t)
        cnt = 0

        if regex == "sigmoid":
            for i in range(size_t):
                if t[i].item() >= self.threshold_sigmoid_max or t[i].item() <= self.threshold_sigmoid_min:
                    cnt += 1

        if regex == "tanh":
            for i in range(size_t):
                if t[i].item() >= self.threshold_tanh_max or t[i].item() <= self.threshold_tanh_min:
                    cnt += 1

        return float(cnt/size_t)

    def invoke_at_step(self, cur_step, layer_names, rule_id, last_step=None):
        if rule_id == 17:
            step_relus = list()
            for tname in layer_names:
                last_tensor = self.base_trial.tensor(tname).value(step_num=last_step, mode=smd.modes.TRAIN)
                cur_tensor = self.base_trial.tensor(tname).value(step_num=cur_step, mode=smd.modes.TRAIN)
                percent = self.compute_dying_relus(last_tensor, cur_tensor)
                if percent >= self.threshold_layer:
                    step_relus.append([tname, percent, True])
                else :
                    step_relus.append([tname, percent, False])
            return step_relus

        if rule_id == 15:
            step_tensors = list()
            for tname in layer_names:
                tensor = self.base_trial.tensor(tname).value(step_num=cur_step, mode=smd.modes.TRAIN)
                percent = self.compute_saturation("sigmoid", tensor)
                if percent >= self.threshold_layer:
                    step_tensors.append([tname, percent, True])
                else :
                    step_tensors.append([tname, percent, False])
            return step_tensors

        if rule_id == 16:
            step_tensors = list()
            for tname in layer_names:
                tensor = self.base_trial.tensor(tname).value(step_num=cur_step, mode=smd.modes.TRAIN)
                percent = self.compute_saturation("tanh", tensor)
                if percent >= self.threshold_layer:
                    step_tensors.append([tname, percent, True])
                else :
                    step_tensors.append([tname, percent, False])
            return step_tensors
    
    def sigmoid_saturation(self):
        path = self.path + '/Sigmoidsaturation'
        if not os.path.exists(path):
            os.makedirs(path)

        layer_names = self.get_output_layer_names(regex="sigmoid")
        percents = {'layers':[], 'percents':[]}
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()
           
        for step in self.steps:
            percents['layers'].clear()
            percents['percents'].clear()
            step_sigs = self.invoke_at_step(cur_step=step, 
                                             layer_names=layer_names, rule_id=15) 
            for step_sig in step_sigs:
                percents['layers'].append(step_sig[0])
                percents['percents'].append(step_sig[1])
                if step_sig[2] == True and step != self.steps[0]:
                    # print(step_zero[0], ": Sigmoid saturation")
                    self.epoch_info['tanh/sigmoid_saturation'] = True
                    update_epochfile(self.epoch_info)
                    results[step_sig[0]].append(1)

                else:
                    # print(step_zero[0], ": Normal Sigmoid")
                    results[step_sig[0]].append(0)

            self.make_data_chart(step, percents, path)

        self.make_result_chart(self.steps, results, path)

    def tanh_saturation(self):
        path = self.path + '/Tanhsaturation'
        if not os.path.exists(path):
            os.makedirs(path)

        layer_names = self.get_output_layer_names(regex="tanh")
        percents = {'layers':[], 'percents':[]}
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()
          
        for step in self.steps:
            percents['layers'].clear()
            percents['percents'].clear()
            step_zeros = self.invoke_at_step(cur_step=step, 
                                             layer_names=layer_names, rule_id=16) 
            for step_tanh in step_zeros:
                percents['layers'].append(step_tanh[0])
                percents['percents'].append(step_tanh[1])
                if step_tanh[2] == True and step != self.steps[0]:
                    # print(step_zero[0], ": Tanh saturation")
                    self.epoch_info['tanh/sigmoid_saturation'] = True
                    update_epochfile(self.epoch_info)
                    results[step_tanh[0]].append(1)
                else:
                    # print(step_zero[0], ": Normal Tanh")
                    results[step_tanh[0]].append(0)

            self.make_data_chart(step, percents, path)

        self.make_result_chart(self.steps, results, path)

    def dying_relu(self):
        path = self.path + '/Dyingrelu'
        if not os.path.exists(path):
            os.makedirs(path)

        layer_names = self.get_output_layer_names(regex="relu")
        percents = {'layers':[], 'percents':[]}
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()

        cnt = 0
        last_step = self.steps[0]
        for step in self.steps: 
            percents['layers'].clear()
            percents['percents'].clear()
            step_tensors = self.invoke_at_step(last_step=last_step, cur_step=step, 
                                              layer_names = layer_names, rule_id=17) 
            for step_relu in step_tensors:
                percents['layers'].append(step_relu[0])
                percents['percents'].append(step_relu[1])
                if step_relu[2] == True and step != self.steps[0]:
                    # print(step_var[0], ": Dying relu")
                    self.epoch_info['dead_relu'] = True
                    update_epochfile(self.epoch_info)
                    results[step_relu[0]].append(1)
                else :
                    # print(step_var[0], ": Normal relu")
                    results[step_relu[0]].append(0)
            last_step = step
 
            self.make_data_chart(step, percents, path)

        self.make_result_chart(self.steps, results, path)

def load_epochfile():
    with open("./debug_info/epoch_info.json",'r') as load_f:
        epoch_info = json.load(load_f)
    return epoch_info

def update_epochfile(epoch_info):
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epoch_info, f)

