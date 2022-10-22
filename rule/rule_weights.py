from array import array
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import os
import json

class Rule_Weights():
    def __init__(self, base_trial, threshold=2.0, small_threshold=0.01):
        super().__init__()
        self.base_trial = base_trial
        self.threshold = float(threshold)
        self.small_threshold = float(small_threshold)
        self.path = './debug_info/weights'
        self.epoch_info = load_epochfile()

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def invoke_at_step(self, cur_step, rule_id, last_step=None):
        list_r = []
        if rule_id==8:
            for tname in self.base_trial.tensor_names(collection="weights", mode=smd.modes.TRAIN):
                t = self.base_trial.tensor(tname)
                var = t.reduction_value(cur_step, "variance", mode=smd.modes.TRAIN)
                list_r.append((tname,var))
            return list_r
        if rule_id==9:
            for tname in self.base_trial.tensor_names(collection="weights", mode=smd.modes.TRAIN):
                t = self.base_trial.tensor(tname)
                t_w = t.value(step_num=cur_step, worker=None, mode=smd.modes.TRAIN)
                list_r.append(t_w)
            return list_r

    def poor_initialization(self):
        steps = self.base_trial.steps(mode=smd.modes.TRAIN)
        list_all = []
        for i in steps:
            thelist = self.invoke_at_step(cur_step=i,rule_id=8)
            if i==0:
                len_sort = len(thelist)
            list_all = list_all+thelist
        len_weights = len_sort*2
        len_steps = len(steps)
        array_weights = np.array(list_all)
        array_weights = array_weights.reshape(len_steps,len_weights)
        list_poor = []
        for i in range(len_sort):
            s = i*2+1
            min = array_weights[0,s]
            max = array_weights[0,s]
            for j in range(len_steps):
                if array_weights[j,s]>max:
                    max = array_weights[j,s]
                if array_weights[j,s]<min:
                    min = array_weights[j,s]
            max = float(max)
            min = float(min)
            ratio = max/min
            if ratio >self.threshold:
                self.epoch_info['poor_initialization'] = True
                update_epochfile(self.epoch_info)
                list_poor.append(array_weights[0,s-1])
        if len(list_poor)!=0:
            df = pd.DataFrame(list_poor)
            df.to_csv(self.path+'/PoorResult.csv')
        df = pd.DataFrame(array_weights)
        df.to_csv(self.path+'/PoorInitialization.csv')

    def update_too_small(self):

        steps = self.base_trial.steps(mode=smd.modes.TRAIN)
        start = 0
        list_up = []
        x = []
        list_small = []
        for i in steps:
            list_w = self.invoke_at_step(cur_step=i,rule_id=9)
            array_sum = list_w[0]
            array_sum = array_sum.reshape(-1)
            for j in range(len(list_w)):
                if j!=0:
                    arrj = list_w[j]
                    arrj = arrj.reshape(-1)
                    array_sum = np.concatenate((array_sum,arrj))
            len_data = len(array_sum)
            if start==0:
                array_old = array_sum
            else:
                sum = 0
                for k in range(len_data):
                    sum = sum+(array_sum[k]-array_old[k])/array_old[k]
                mu = abs(sum/len_data)
                list_up.append(mu)
                if mu<self.small_threshold:
                    self.epoch_info['update_small'] = True
                    update_epochfile(self.epoch_info)
                    list_small.append((i,mu))
            start = start+1
            x.append(start)
        while(len(x)>len(list_up)):
            x.pop()
        while(len(x)<len(list_up)):
            list_up.pop()
        #dict_update = {'step':x,'update':list_up}
        dict_update = {'update':list_up}
        df = pd.DataFrame(dict_update)
        df.to_csv(self.path+'/UpdateTooSmall.csv')
        if len(list_small)==0:
            list_steps = ['None']
            dict_small = {'step':list_steps,'update_ratio':'None'}
            df2 = pd.DataFrame(dict_small)
            df2.to_csv(self.path+'/UpdateResult.csv')
        if len(list_small)!=0:
            list_steps = []
            list_ratio = []
            for i in list_small:
                list_steps.append(i[0])
                list_ratio.append(i[1])
            dict_small = {'step':list_steps,'update_ratio':list_ratio}
            df2 = pd.DataFrame(dict_small)
            df2.to_csv(self.path+'/UpdateResult.csv')

def load_epochfile():
    with open("./debug_info/epoch_info.json",'r') as load_f:
        epoch_info = json.load(load_f)
    return epoch_info

def update_epochfile(epoch_info):
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epoch_info, f)