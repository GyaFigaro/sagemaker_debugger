import streamlit as st
import time
import numpy as np
import os
import pandas as pd
import json

show_step = 5

st.markdown("# Tensor规则")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.markdown("## Rule: 张量值全零")

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
    stats = "step " + str(num) + " 中各张量0值百分比:"
    st.markdown(stats)
    st.bar_chart(df2, x="layers", y="percents")

st.markdown("## 张量值未变")

df1 = pd.read_csv('./debug_info/tensor/ValuesUnchanged/result.csv')
df1.set_index("step")

for index in list(df1.index.values):
    flag = 0
    step = df1.iloc[index]['step']
    stats = "step " + str(step) + "中，张量值未变的层为："
    st.markdown(stats)
    for lname, item in df1.iloc[index].iteritems():
        if item == 1:
            st.markdown(lname)
            flag = 1
    if flag == 0:
        st.markdown("无")
