from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
from smdebug.rules.rule import Rule

import numpy as np
import matplotlib.pyplot as plt

class AllValuesZeroRule():
    def __init__(self, base_trial, threshold_layer=1):
        super().__init__()
        self.threshold_layer = float(threshold_layer)
        self.base_trial = base_trial

    def invoke_at_step(self, step, layer_names):
        step_zeros = list()
        for tname in layer_names:
            tensor = self.base_trial.tensor(tname).value(step)
            percent = self.compute_zero_values(tensor)
            if percent >= self.threshold_layer:
                step_zeros.append([tname, percent, True])
            else :
                step_zeros.append([tname, percent, False])
        return step_zeros

    def compute_zero_values(self, tensor):
        t = tensor.reshape(-1)
        size_t = np.size(t)
        cnt = 0
        for i in range(size_t):
            if t[i].item() == 0:
                cnt += 1
        return float(cnt/size_t)

    def get_output_layer_names(self):
        layer_names = list()
        for tname in self.base_trial.tensor_names(collection="all"):
            if tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names
    
    def draw_plot(self, steps, layer_percents):
        plt.figure(figsize=(20,8), dpi=90)
        for key in layer_percents:
            print(key)
            plt.figure(figsize=(20,8), dpi=90)
            plt.plot(steps, layer_percents[key], lw=2, ls='-', c='r', alpha=0.1)
            plt.show() 

    def work(self):
        steps = self.base_trial.steps()
        layer_names = self.get_output_layer_names()
        layer_percents = dict()
        for lname in layer_names:
            percents = list()
            layer_percents[lname] = percents               
        for step in steps:
            print("step ", step, ":")
            step_zeros = self.invoke_at_step(step, layer_names) 
            for step_zero in step_zeros:
                layer_percents[step_zero[0]].append(step_zero[1])
                if step_zero[2] == True and step != steps[0]:
                    print(step_zero[0], ": All values zero")
                else: print(step_zero[0], ": Not all values zero")
        self.draw_plot(steps, layer_percents)

class SmallVarianceRule():
    def __init__(self, base_trial, min_threshold=0.0001):
        super().__init__()
        self.min_threshold = float(min_threshold)
        self.base_trial = base_trial

    def invoke_at_step(self, step):
        step_vars = list()
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            is_small = False
            if var < self.min_threshold:
                is_small = True
            step_var = (tname, var, is_small)
            step_vars.append(step_var)
        return step_vars
    
    def work(self):
        steps = self.base_trial.steps()
        for step in steps:
            print("step ", step, ":")
            step_vars = self.invoke_at_step(step)
            layer_names = list(step_vars[i][0] for i in range(4))
            vars = list(step_vars[i][1] for i in range(4))
            plt.bar(layer_names, vars)
            plt.show()      
            for step_var in step_vars:
                if step_var[2] == True:
                    print(step_var[0], ": Variance of values is too small")


class ValuesUnchangedRule():
    def __init__(self, base_trial, rtol=1e-01, atol=1e-01):
        super().__init__()
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.base_trial = base_trial

    def invoke_at_step(self, last_step, cur_step):
        step_tensors = list()
        for tname in self.base_trial.tensor_names(collection="weights"):
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
    
    def work(self):
        steps = self.base_trial.steps()
        last_step = steps[0]
        for step in steps: 
            print("step ", step, ":")
            step_tensors = self.invoke_at_step(last_step, step) 
            for step_var in step_tensors:
                if step_var[1] == True and step != steps[0]:
                    print(step_var[0], ": Tensors were unchanged")
                print(step_var[0], ": Tensors changed properly")
            last_step = step


class DeadReluRule():
    def __init__(self, base_trial, threshold_layer=0.4):
        super().__init__()
        # self.threshold_inactivity = float(threshold_inactivity)
        self.threshold_layer = float(threshold_layer)
        self.base_trial = base_trial

    def invoke_at_step(self, last_step, cur_step, layer_names):
        step_relus = list()
        for tname in layer_names:
            last_tensor = self.base_trial.tensor(tname).value(last_step)
            cur_tensor = self.base_trial.tensor(tname).value(cur_step)
            percent = self.compute_dying_relus(last_tensor, cur_tensor)
            if percent >= self.threshold_layer:
                step_relus.append([tname, percent, True])
            else :
                step_relus.append([tname, percent, False])
        return step_relus
    
    def compute_dying_relus(self, last_tensor, cur_tensor):
        last_t = last_tensor.reshape(-1)
        cur_t = cur_tensor.reshape(-1)
        size_t = np.size(last_t)
        cnt = 0
        for i in range(size_t):
            if last_t[i].item() == 0 and cur_t[i].item() == 0:
                cnt += 1
        return float(cnt/size_t)

    def get_relu_layer_names(self):
        layer_names = list()
        for tname in self.base_trial.tensor_names(collection="all"):
            if tname.find('relu') != -1 and tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names

    def draw_plot(self, steps, layer_percents):
        plt.figure(figsize=(20,8), dpi=90)
        for key in layer_percents:
            print(key)
            plt.figure(figsize=(20,8), dpi=90)
            plt.plot(steps, layer_percents[key], lw=2, ls='-', c='r', alpha=0.1)
            plt.show() 

    def work(self):
        steps = self.base_trial.steps()
        last_step = steps[0]
        layer_names = self.get_relu_layer_names()
        layer_percents = dict()
        for lname in layer_names:
            percents = list()
            layer_percents[lname] = percents               
        for step in steps:
            print("step ", step, ":")
            step_relus = self.invoke_at_step(last_step, step, layer_names) 
            for step_relu in step_relus:
                layer_percents[step_relu[0]].append(step_relu[1])
                if step_relu[2] == True and step != steps[0]:
                    print(step_relu[0], ": Dying relu")
                else: print(step_relu[0], ": Normal relu")
            last_step = step
        self.draw_plot(steps, layer_percents)


class SigmondSaturationRule():
    def __init__(self, base_trial, threshold_gradients=0.0001, threshold_layer=0.6):
        super().__init__()
        self.threshold_gradients = float(threshold_gradients)
        self.threshold_layer = float(threshold_layer)
        self.base_trial = base_trial

    def invoke_at_step(self, last_step, cur_step, layer_names):
        step_sigs = list()
        for tname in layer_names:
                last_tensor = self.base_trial.tensor(tname).value(last_step)
                cur_tensor = self.base_trial.tensor(tname).value(cur_step)
                percent = self.compute_dying_sigs(last_tensor, cur_tensor)
                if percent >= self.threshold_layer:
                    step_sigs.append([tname, percent, True])
                else :
                    step_sigs.append([tname, percent, False])
        return step_sigs
    
    def compute_dying_sigs(self, last_tensor, cur_tensor):
        last_t = last_tensor.reshape(-1)
        cur_t = cur_tensor.reshape(-1)
        size_t = np.size(last_t)
        cnt = 0
        for i in range(size_t):
            if last_t[i].item() - cur_t[i].item() <= self.threshold_gradients:
                cnt += 1
        return float(cnt/size_t)

    def get_sig_layer_names(self):
        layer_names = list()
        for tname in self.base_trial.tensor_names(collection="all"):
            if tname.find('sigmoid') != -1 and tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names

    def draw_plot(self, steps, layer_percents):
        plt.figure(figsize=(20,8), dpi=90)
        for key in layer_percents:
            print(key)
            plt.figure(figsize=(20,8), dpi=90)
            plt.plot(steps, layer_percents[key], lw=2, ls='-', c='r', alpha=0.1)
            plt.show() 

    def work(self):
        steps = self.base_trial.steps()
        last_step = steps[0]
        layer_names = self.get_sig_layer_names()
        layer_percents = dict()
        for lname in layer_names:
            percents = list()
            layer_percents[lname] = percents               
        for step in steps:
            print("step ", step, ":")
            step_sigs = self.invoke_at_step(last_step, step, layer_names) 
            for step_sig in step_sigs:
                layer_percents[step_sig[0]].append(step_sig[1])
                if step_sig[2] == True and step != steps[0]:
                    print(step_sig[0], ": Sigmoid saturation")
                else: print(step_sig[0], ": Normal Sigmoid")
            last_step = step
        self.draw_plot(steps, layer_percents)

class TanhSaturationRule():
    def __init__(self, base_trial, threshold_gradients=0.0001, threshold_layer=0.4):
        super().__init__()
        self.threshold_gradients = float(threshold_gradients)
        self.threshold_layer = float(threshold_layer)
        self.base_trial = base_trial

    def invoke_at_step(self, last_step, cur_step, layer_names):
        step_tanhs = list()
        for tname in layer_names:
                last_tensor = self.base_trial.tensor(tname).value(last_step)
                cur_tensor = self.base_trial.tensor(tname).value(cur_step)
                percent = self.compute_dying_sigs(last_tensor, cur_tensor)
                if percent >= self.threshold_layer:
                    step_tanhs.append([tname, percent, True])
                else :
                    step_tanhs.append([tname, percent, False])
        return step_tanhs
    
    def compute_dying_tanhs(self, last_tensor, cur_tensor):
        last_t = last_tensor.reshape(-1)
        cur_t = cur_tensor.reshape(-1)
        size_t = np.size(last_t)
        cnt = 0
        for i in range(size_t):
            if last_t[i].item() - cur_t[i].item() <= self.threshold_gradients:
                cnt += 1
        return float(cnt/size_t)

    def get_tanh_layer_names(self):
        layer_names = list()
        for tname in self.base_trial.tensor_names(collection="all"):
            if tname.find('tanh') != -1 and tname.find('output') != -1:
                layer_names.append(tname)
        return layer_names

    def draw_plot(self, steps, layer_percents):
        plt.figure(figsize=(20,8), dpi=90)
        for key in layer_percents:
            print(key)
            plt.figure(figsize=(20,8), dpi=90)
            plt.plot(steps, layer_percents[key], lw=2, ls='-', c='r', alpha=0.1)
            plt.show() 

    def work(self):
        steps = self.base_trial.steps()
        last_step = steps[0]
        layer_names = self.get_tanh_layer_names()
        layer_percents = dict()
        for lname in layer_names:
            percents = list()
            layer_percents[lname] = percents               
        for step in steps:
            print("step ", step, ":")
            step_tanhs = self.invoke_at_step(last_step, step, layer_names) 
            for step_tanh in step_tanhs:
                layer_percents[step_tanh[0]].append(step_tanh[1])
                if step_tanh[2] == True and step != steps[0]:
                    print(step_tanh[0], ": Sigmond saturation")
                else: print(step_tanh[0], ": Normal Sigmond")
            last_step = step
        self.draw_plot(steps, layer_percents)