import time
from unittest import result
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os
from urllib.error import URLError
import json

def overfitting():
    data = pd.read_csv('./debug_info/data4.csv')
    st.line_chart(data, x="steps")

def all_values_zero(show_step):
    df1 = pd.read_csv('./debug_info/tensor/AllZeroValues/result.csv')
    df1.set_index("step")
    step_list = list()
    for index in list(df1.index.values):
        flag = 0
        step = df1.iloc[index]['step']
        stats = "step" + str(step) + "中, 张量值为0的层为: "
        st.markdown(stats)
        for lname, item in df1.iloc[index].iteritems():
            if item == 1:
                st.markdown(lname)
                flag = 1
        if flag == 0:
            st.markdown("无")
        else:
            step_list.append(step)
    if len(step_list) < show_step:
        show_step = len(step_list)
    for num in step_list[:show_step]:
        path = './debug_info/tensor/AllZeroValues/' + str(num) + '.csv'
        df2 = pd.read_csv(path)
        stats = "step" + str(num) + " 中各张量0值百分比:"
        st.markdown(stats)
        st.bar_chart(df2, x="layers", y="percents")

def tensor_unchanged():
    df1 = pd.read_csv('./debug_info/tensor/ValuesUnchanged/result.csv')
    df1.set_index("step")
    for index in list(df1.index.values):
        flag = 0
        step = df1.iloc[index]['step']
        stats = "step" + str(step) + "中，张量值未变的层为："
        st.markdown(stats)
        for lname, item in df1.iloc[index].iteritems():
            if item == 0:
                st.markdown(lname)
                flag = 1
        if flag == 0:
            st.markdown("无")

def dead_relu(show_step):
    df1 = pd.read_csv('./debug_info/activationfunction/Dyingrelu/result.csv')
    df1.set_index("step")
    step_list = list()
    for index in list(df1.index.values):
        flag = 0
        step = df1.iloc[index]['step']
        stats = "step" + str(step) + "中, relu失活的层为: "
        st.markdown(stats)
        for lname, item in df1.iloc[index].iteritems():
            if item == 0:
                st.markdown(lname)
                flag = 1
        if flag == 0:
            st.markdown("无")
        else:
            step_list.append(step)
    if len(step_list) < show_step:
        show_step = len(step_list)
    for num in step_list[:show_step]:
        path = './debug_info/activationfunction/Dyingrelu/' + str(num) + '.csv'
        df2 = pd.read_csv(path)
        stats = "step" + str(num) + " 中各relu失活百分比为:"
        st.markdown(stats)
        st.bar_chart(df2, x="layers", y="percents")

def tanh_sig_saturation(show_step):
    df1 = pd.read_csv('./debug_info/activationfunction/Tanhsaturation/result.csv')
    df2 = pd.read_csv('./debug_info/activationfunction/Sigmoidsaturation/result.csv')
    df3 = pd.concat([df1,df2], axis=1, join='inner')
    print(df3)
    step_list = list()
    for index in list(df3.index.values):
        flag = 0
        step = df3.iloc[index]['step']
        stats = "step" + str(step) + "中, 饱和的tanh/sigmoid层为:"
        st.markdown(stats)
        for lname, item in df3.iloc[index].iteritems():
            if item == 0:
                st.markdown(lname)
                flag = 1
        if flag == 0:
            st.markdown("无")
        else:
            step_list.append(step)
    if len(step_list) < show_step:
        show_step = len(step_list)
    for num in step_list[:show_step]:
        path1 = './debug_info/activationfunction/Tanhsaturation/' + str(num) + '.csv'
        df4 = pd.read_csv(path1)
        path2 = './debug_info/activationfunction/Sigmoidsaturation/' + str(num) + '.csv'
        df5 = pd.read_csv(path2)
        df6 = pd.concat([df4, df5], axis = 0)
        stats = "step" + str(num) + " 中各tanh/sigmoid层饱和百分比为: "
        st.markdown(stats)
        st.bar_chart(df6, x="layers", y="percents")
    



