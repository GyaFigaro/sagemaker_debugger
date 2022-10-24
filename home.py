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

def show():
    f = open('./debug_info/result1.txt', 'r')
    result1 = f.read()
    f.close()
    if result1 == "True":
        st.info("""数据集不平衡:   未检出""")
    else:
        st.info("""数据集不平衡：  已检测到该问题""")
    f = open('./debug_info/result2.txt', 'r')
    result = f.read()
    f.close()
    if result == "0":
        st.info("""数据未均一化:   未检出""")
    elif result == '1':
        st.info("""数据未均一化:   已检测到该问题""")
    else:
        st.info("""数据未均一化:   数据量过少无法检测""")
    f = open('./debug_info/result3.txt', 'r')
    result = f.read()
    f.close()
    if result == "True":
        st.info("""损失不减少:   未检出""")
    else:
        st.info("""损失不减少：  已检测到该问题""")
    f = open('./debug_info/result4.txt', 'r')
    result = f.read()
    f.close()
    if result == "True":
        st.info("""过拟合:   未检出""")
    else:
        st.info("""过拟合：  已检测到该问题""")
    f = open('./debug_info/result6.txt', 'r')
    result = f.read()
    f.close()
    if result == "True":
        st.info("""欠拟合:   未检出""")
    else:
        st.info("""欠拟合：  已检测到该问题""")
    if os.path.exists('./debug_info/result5.txt') == True:
        f = open('./debug_info/result5.txt', 'r')
        result = f.read()
        f.close()
        if result == "True":
            st.info("""模型预测准确率:   良好""")
        else:
            st.info("""模型预测准确率：  未达标""")

# 加载每个rule的结果
def load_result(epoch_info, show_step):
    st.info("详细信息")
    show()
    for key in epoch_info:

        # 按rule名，对每条rule的结果进行展示
        if key == 'poor_initialization':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('初始化不当:   已检测到该问题')
            else:
                st.info('初始化不当:   未检出')
        if key == 'update_small':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('张量更新过慢:   已检测到该问题')
            else:
                st.info('张量更新过慢:   未检出')
        if key == 'vanishing_gradient':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('梯度消失:   已检测到该问题')
            else:
                st.info('梯度消失:   未检出')
        if key == 'exploding_gradient':
            if epoch_info[key] == True:   # true 为检测到问题，false相反
                st.info('梯度爆炸:   已检测到该问题')
            else:
                st.info('梯度爆炸:   未检出')
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
                    time.sleep(5)
