import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import os
import json

class Rule_Gradients():
    def __init__(self, base_trial, vanishing_threshold=0.0000001, exploding_threshold=1):
        super().__init__()
        self.base_trial = base_trial
        self.vanishing_threshold = float(vanishing_threshold)
        self.exploding_threshold = float(exploding_threshold)
        self.path = './debug_info/gradients'
        self.epoch_info = load_epochfile()

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def invoke_at_step(self, cur_step):
        list_gradients = []
        for tname in self.base_trial.tensor_names(collection="gradients", mode=smd.modes.TRAIN):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(cur_step, "mean", mode=smd.modes.TRAIN, worker=None, abs=True)
            list_gradients.append(abs_mean)
        return list_gradients

    def gradients(self):
        steps = self.base_trial.steps(mode=smd.modes.TRAIN)
        list_gradients = []
        for i in steps:
            list_gradients = list_gradients + self.invoke_at_step(i)
        len_steps = len(steps)
        len_gradients = int(len(list_gradients)/len_steps)
        array_gradients = np.array(list_gradients)
        array_gradients = array_gradients.reshape(len_steps, len_gradients)
        df = pd.DataFrame(array_gradients)
        df.to_csv(self.path+'/gradients.csv')
        bool_vanishing = False
        bool_exploding = False
        for i in list_gradients:
            if i < self.vanishing_threshold:
                self.epoch_info['vanishing_gradient'] = True
                update_epochfile(self.epoch_info)
                bool_vanishing = True
            if i > self.exploding_threshold:
                self.epoch_info['exploding_gradient'] = True
                update_epochfile(self.epoch_info)
                bool_exploding = True
        array_result = np.array(['vanishing',bool_vanishing,'exploding',bool_exploding])
        array_result = array_result.reshape(2,2)
        df2 = pd.DataFrame(array_result)
        df2.to_csv(self.path+'/GradientsResult.csv')

def load_epochfile():
    with open("./debug_info/epoch_info.json",'r') as load_f:
        epoch_info = json.load(load_f)
    return epoch_info

def update_epochfile(epoch_info):
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epoch_info, f)