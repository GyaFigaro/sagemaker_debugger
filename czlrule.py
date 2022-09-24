import collections
from crypt import methods
from itertools import accumulate
from statistics import mean, variance
from typing import Collection, Concatenate
from typing_extensions import Self
from xmlrpc.client import boolean
from crypt import methods
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
import seaborn as sns
import streamlit as st
import pandas as pd

class PoorInitialization():
    def __init__(self, base_trial, threshold=10.0):
        super().__init__()
        self.base_trial = base_trial
        self.threshold = float(threshold)
    def invoke_at_step(self, step):
        listvar = []
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            listvar.append((tname,var))
        maxvar = listvar[0][1]
        minvar = listvar[0][1]
        maxname = listvar[0][0]
        minname = listvar[0][0]
        for var_t in listvar:
            if var_t[1] > maxvar:
                maxvar = var_t[1]
                maxname = var_t[0]
            if var_t[1] < minvar:
                minvar = var_t[1]
                minname = var_t[0]
        ratio = maxvar/minvar
        listm = [(minname, minvar), (maxname, maxvar),ratio]
        return listm
    def work(self):
        steps = self.base_trial.steps()
        vardict = {}
        for i in steps:
            thelist = self.invoke_at_step(i)
            vardict[i] = thelist
        x = []
        ratio = []
        lista = []
        for i in steps:
            x.append(i)
            listi = vardict[i]
            ratio.append(listi[2])
            if ratio[-1] > self.threshold:
                lista.append(i)
        #plt.plot(x,ratio,linewidth=1, color="orange", marker="o",label="variance ratio")
        #plt.show()
        st.title('PoorInitialization')
        fig, ax = plt.subplots()
        ax.plot(x,ratio,linewidth=1, color="orange", marker="o",label="variance ratio")

        st.pyplot(fig)
        if len(lista)!=0:
            st.write("Weight Poor Initialization")
            for i in lista:
                st.write("step ",i," ratio: ",vardict[i][2])

class UpdatesTooSmall():
    def __init__(self, base_trial, small_threshold=10.0):
        super().__init__()
        self.base_trial = base_trial
        self.small_threshold = float(small_threshold)
    def invoke_at_step(self, step):
        list_w = []
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            #t_w = t.reduction_value(step,"l2",abs=True)
            t_w = t.value(step, worker=None)
            #t_w = t.shape(step)
            list_w.append(t_w)
        return list_w
    def work(self):
        st.title('UpdatesTooSmall')
        steps = self.base_trial.steps()
        start = 0
        list_up = []
        x = []
        alen = int(430500)
        for i in steps:
            list_w = self.invoke_at_step(i)
            arr1 = list_w[0]
            arr2 = list_w[1]
            arr3 = list_w[2]
            arr4 = list_w[3]
            arr1 = arr1.reshape(-1)
            arr2 = arr2.reshape(-1)
            arr3 = arr3.reshape(-1)
            arr4 = arr4.reshape(-1)
            newarr = np.concatenate((arr1,arr2))
            newarr = np.concatenate((newarr,arr3))
            newarr = np.concatenate((newarr,arr4))
            if start==0:
                oldarr = newarr
            else:
                sum = 0
                for j in range(alen):
                    sum = sum+(newarr[j]-oldarr[j])/oldarr[j]
                mu = sum/alen
                list_up.append(mu)
            start = start+1
            x.append(start)
        x.pop()
        fig, ax = plt.subplots()
        ax.plot(x,list_up,linewidth=1, color="orange", marker="o",label="weight_update_ratio")

        st.pyplot(fig)
        #plt.plot(x,list_up,linewidth=1, color="orange", marker="o",label="weight_update_ratio")
        #plt.show()
        
        
class Gradients_Vanishing():
    def __init__(self, base_trial, vanishing_threshold=0.0000001):
        super().__init__()
        self.base_trial = base_trial
        self.vanishing_threshold = float(vanishing_threshold)
    def invoke_at_step(self, step):
        list_gradients = []
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step,"mean",worker=None,abs=True)
            list_gradients.append(abs_mean)
        return list_gradients
    def work(self):
        steps = self.base_trial.steps()
        s = 0
        for i in steps:
            list_gradients = []
            list_gradients = self.invoke_at_step(i)
            arr1 = np.array(list_gradients)
            if s!=0:
                arrs = np.concatenate((arrs,arr1))
            else:
                arrs = np.array(list_gradients)
            s = s+1
            van = False
            for g in list_gradients:
                if g < self.vanishing_threshold:
                    van = True
            if van:
                l = len(list_gradients)
                list_l = []
                for j in range(l):
                    list_l.append(j+1)
                print("Step ",i," Gradients Vanishing")
                #plt.plot(list_l,list_gradients,linewidth=1, color="orange", marker="o",label="gradients")
                #plt.show()
        r = len(steps)
        c = len(arrs)/r
        c = int(c)
        newarr = arrs.reshape(r,c)
        sns.set_theme()
        ax = sns.heatmap(newarr,annot=True,fmt=".4f",linewidths=.9000)
        plt.savefig("./image/vanishing")
        st.title('Gradients_Vanishing')
        st.image("./image/vanishing.png")

class Gradients_Exploding():
    def __init__(self, base_trial, exploding_threshold=1):
        super().__init__()
        self.base_trial = base_trial
        self.exploding_threshold = float(exploding_threshold)
    def invoke_at_step(self, step):
        list_gradients = []
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step,"mean",worker=None,abs=True)
            list_gradients.append(abs_mean)
        return list_gradients
    def work(self):
        steps = self.base_trial.steps()
        s = 0
        for i in steps:
            list_gradients = []
            list_gradients = self.invoke_at_step(i)
            arr1 = np.array(list_gradients)
            if s!=0:
                arrs = np.concatenate((arrs,arr1))
            else:
                arrs = np.array(list_gradients)
            s = s+1
            exp = False
            for g in list_gradients:
                if g > self.exploding_threshold:
                    exp = True
            if exp:
                l = len(list_gradients)
                list_l = []
                for j in range(l):
                    list_l.append(j+1)
                print("Step ",i," Gradients Exploding")
                plt.plot(list_l,list_gradients,linewidth=1, color="orange", marker="o",label="gradients")
                plt.show()
        r = len(steps)
        c = len(arrs)/r
        c = int(c)
        newarr = arrs.reshape(r,c)
        sns.set_theme()
        ax = sns.heatmap(newarr,annot=True,fmt=".4f",linewidths=.9000)
        st.title('radients_Exploding')
        plt.savefig("./image/exploding")
        st.image("./image/exploding.png")


trial = smd.create_trial(path="./tmp2/testing/demo/")
rule_weight = PoorInitialization(base_trial=trial,threshold=10.0)
rule_weight.work()


rule_update = UpdatesTooSmall(base_trial=trial,small_threshold=10.0)
rule_update.work()


rule_gradients = Gradients_Vanishing(base_trial=trial,vanishing_threshold=0.001)
rule_gradients.work()


rule_gradients = Gradients_Exploding(base_trial=trial,exploding_threshold=0.5)
rule_gradients.work()
#app = Flask(__name__)
#@app.route('/')
#def index():
    #rule_weight = PoorInitialization(base_trial=trial,threshold=10.0)
    #rule_weight.work()
    #return render_template('seven.html'
    #return "hello world"

#if __name__=='__main__':
#    app.run()
