import time
from unittest import result
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os
from urllib.error import URLError
import json
from display_funcs import overfitting, all_values_zero, tensor_unchanged, dead_relu, tanh_sig_saturation

show_step = 5

def get_result(epoch_info):
    for key in epoch_info:
        if key != 'epoch_num':
            if epoch_info[key] == True:
                return False
    return True

# 加载每个rule的结果
def load_result(epoch_info, show_step):
    st.info("详细信息")
    for key in epoch_info:

        # 按rule名，对每条rule的结果进行展示

        if key == 'overfitting':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('过拟合:   已检测到该问题')
            else:
                st.info('过拟合:   未检出')
        if key == 'underfitting':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('过拟合:   已检测到该问题')
            else:
                st.info('过拟合:   未检出')
        if key == 'all_values_zero':
            if epoch_info[key] == True:
                st.info('张量值全零:   已检测到该问题')
            else:
                st.info('张量值全零:   未检测出')
        if key == 'tensors_unchanged':
            if epoch_info[key] == True:
                st.info('张量值未变:   已检测到该问题')
            else:
                st.info('张量值全零:   未检测出')
        if key == 'dead_relu':
            if epoch_info[key] == True:
                st.info('失活relu:   已检测到该问题')
            else:
                st.info('失活relu:   未检测出')
        if key == 'tanh/sigmoid_saturation':
            if epoch_info[key] == True:
                st.info('tanh/sigmoid饱和:   已检测到该问题')
            else:
                st.info('tanh/sigmoid饱和:   未检测出')


st.set_page_config(page_title="ML Debug Demo", page_icon="📊")

st.markdown("# 调试结果")
st.markdown(" ")
st.markdown(" ")

last_epoch = -1

while True:
    if os.path.exists("./debug_info/epoch_info.json") is False:
        with st.spinner("Loading..."):
            time.sleep(5)
    else:
        with open("./debug_info/epoch_info.json",'r') as load_f:
            load_epo = json.load(load_f)
            epoch = load_epo['epoch_num']
            if epoch != last_epoch:
                print(load_f)
                if get_result(load_epo):
                    stats = "第" + str(epoch) + "轮:   训练正常"
                    st.success(stats)
                else:
                    stats = "第" + str(epoch) + "轮:   训练出现异常"
                    st.error(stats)
                load_result(load_epo, show_step)
                last_epoch = epoch
            else:
                with st.spinner("Loading..."):
                    time.sleep(1)
