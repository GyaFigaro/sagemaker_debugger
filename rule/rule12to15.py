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

class DeadReluRule(Rule):
    def __init__(self, base_trial, min_threshold=0.0001):
        super().__init__(base_trial)
        self.min_threshold = float(min_threshold)

    def invoke_at_step(self, step):
        step_var = list[tuple]
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            if var < self.min_threshold:
                is_small = True
            step_var.append([tname, var, is_small])
        return False
    

class TanhSaturationRule(Rule):
    def __init__(self, base_trial, min_threshold=0.0001):
        super().__init__(base_trial)
        self.min_threshold = float(min_threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            if var < self.min_threshold:
                return True
        return False


class SigmondSaturationRule(Rule):
    def __init__(self, base_trial, min_threshold=0.0001):
        super().__init__(base_trial)
        self.min_threshold = float(min_threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            if var < self.min_threshold:
                return True
        return False