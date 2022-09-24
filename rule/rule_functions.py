import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rule.utils

class Rule_ActivationFunctions():
    def __init__(self, base_trial, 
                 rtol=1e-05, atol=1e-08, 
                 threshold_layer=0.4, 
                 show_steps=5):
        super().__init__()
        self.base_trial = base_trial
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.threshold_layer = float(threshold_layer)
        self.show_steps = show_steps
        self.path = './debug_info/tensor'
        
        
    def get_output_layer_names(self):
        layer_names = list()
        for tname in self.base_trial.tensor_names(collection="all"):
            if tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names

    def make_plot(self, step, dicts, rule_path):
        plot_path = rule_path + '/' + str(step) + '.png'
        print(plot_path)
        x_label = list(dicts.keys())
        y_label = list(dicts.values())
        plt.figure(figsize=(100,40), dpi=90)
        plt.plot(x_label, y_label, lw=2, ls='-', c='r', alpha=0.1)
        plt.savefig(str(plot_path), dpi=90, bbox_inches = 'tight')
        # plt.show()

    def make_chart(self, steps, dicts, rule_path):
        chart_path = rule_path + '/result.csv'
        df = pd.DataFrame({'step':steps})
        cnt = 1
        for key in dicts:
            df.insert(cnt, key, dicts[key])
            cnt += 1
        df.to_csv(chart_path, index=True, header=True, sep=' ')
    
    def compute_zero_values(self, tensor):
        t = tensor.reshape(-1)
        size_t = np.size(t)
        cnt = 0
        for i in range(size_t):
            if t[i].item() == 0:
                cnt += 1
        return float(cnt/size_t)

    def invoke_at_step(self, cur_step, layer_names, rule_id, last_step=None):
        if rule_id == 12:
            step_zeros = list()
            for tname in layer_names:
                tensor = self.base_trial.tensor(tname).value(cur_step)
                percent = self.compute_zero_values(tensor)
                if percent >= self.threshold_layer:
                    step_zeros.append([tname, percent, True])
                else :
                    step_zeros.append([tname, percent, False])
            return step_zeros

        if rule_id == 13:
            step_tensors = list()
            for tname in layer_names:
                last_t = self.base_trial.tensor(tname).value(last_step)
                cur_t = self.base_trial.tensor(tname).value(cur_step)
                is_unchanged = np.allclose(
                    last_t, cur_t, 
                    rtol=self.rtol, 
                    atol=self.atol,
                    equal_nan=False)
                step_tensor = (tname, is_unchanged)
                step_tensors.append(step_tensor)
            return step_tensors
    
    def all_values_zero(self):
        path = self.path + '/AllZeroValues'
        if not os.path.exists(path):
            os.makedirs(path)

        steps = self.base_trial.steps()
        layer_names = self.base_trial.tensor_names(regex="output")
        percents = dict.fromkeys(layer_names)
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()

        cnt = 0            
        for step in steps:
            is_occur = False
            print("step ", step, ":")
            step_zeros = self.invoke_at_step(cur_step=step, 
                                             layer_names=layer_names, rule_id=12) 
            for step_zero in step_zeros:
                percents[step_zero[0]] = step_zero[1]
                if step_zero[2] == True and step != steps[0]:
                    # print(step_zero[0], ": All values zero")
                    results[step_zero[0]].append("All values zero")
                    is_occur = True
                else:
                    # print(step_zero[0], ": Not all values zero")
                    results[step_zero[0]].append("Not all values zero")

            if is_occur and cnt < self.show_steps: 
                self.make_plot(step, percents, path)
                cnt += 1

        self.make_chart(steps, results, path)

    def values_unchanged(self):
        path = self.path + '/ValuesUnchanged'
        if not os.path.exists(path):
            os.makedirs(path)

        steps = self.base_trial.steps()
        layer_names = self.base_trial.tensor_names(collection="weights")
        results = dict.fromkeys(layer_names)
        for lname in layer_names:
            results[lname] = list()

        last_step = steps[0]
        for step in steps: 
            print("step ", step, ":")
            step_tensors = self.invoke_at_step(last_step=last_step, cur_step=step, 
                                              layer_names = layer_names, rule_id=13) 
            for step_var in step_tensors:
                if step_var[1] == True and step != steps[0]:
                    # print(step_var[0], ": Tensors were unchanged")
                    results[step_var[0]].append("Tensors were unchanged")
                else :
                    # print(step_var[0], ": Tensors changed properly")
                    results[step_var[0]].append("Tensors changed properly")
            last_step = step

        self.make_chart(steps, results, path)



