import time
from unittest import result
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os
from urllib.error import URLError
import json

@st.cache(suppress_st_warning=True, show_spinner=True)
def analysis_show(df):
    flag = 0


@st.cache(suppress_st_warning=True, show_spinner=True)
def epoch_show(epoch):
    epoch_str = 'epoch: ' + str(epoch)
    st.markdown(epoch_str)

def load_result(epoch_info):
    for key in epoch_info:
        if key != 'epoch_num':
            if epoch_info[key] == True:
                return False
            elif epoch_info[key] == 1 or epoch_info[key] == 2:
                return False
    return True

st.set_page_config(page_title="ML Debug Demo", page_icon="📊")

st.markdown("# 调试结果")

# st.sidebar.success("Select a type of rule above.")

last_epoch = -1

while True:
    if os.path.exists("./debug_info/epoch_info.json") is False:
        with st.spinner("Loading..."):
            time.sleep(1)
    else:
        with open("./debug_info/epoch_info.json",'r') as load_f:
            load_epo = json.load(load_f)
            epoch = load_epo['epoch_num']
            if epoch != last_epoch:
                print(load_f)
                if load_result(load_epo):
                    st.success("第{}轮:   训练正常", epoch)
                else:
                    st.error("第{}轮:   训练出现异常", epoch)
                    # should_tell_me_more = st.button('Tell me more')
                    # if should_tell_me_more:
                    #     tell_me_more()
                    #     st.markdown('---')
                    # else:
                    #     st.markdown('---')
                    #     interactive_galaxies(df)
                last_epoch = epoch
            else:
                with st.spinner("Loading..."):
                    time.sleep(1)
    # df1 = pd.read_csv('./debug_info/tensor/data.csv')

    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)

    # # for i in range(1, 101):
    # #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    # #     status_text.text("%i%% Complete" % i)
    # #     chart.add_rows(new_rows)
    # #     progress_bar.progress(i)
    # #     last_rows = new_rows
    # #     time.sleep(0.05)

    # st.markdown("## Rule2: Tensor Not Changed")

    # df3 = pd.read_csv('./debug_info/tensor/ValuesUnchanged/result.csv')

    # analysis_show(df3, epoch)

    # st.markdown("# Tensor Rules")

    # st.write(
    #     """This demo illustrates a combination of plotting and animation with
    # Streamlit. We're generating a bunch of random numbers in a loop for around
    # 5 seconds. Enjoy!"""
    # )

    # st.markdown("## Rule3: Dying Relu")

    # df4 = pd.read_csv('./debug_info/activationfunction/Dyingrelu/result.csv')

    # analysis_show(df4, epoch)

    # st.markdown("## Rule4: Sigmoid Saturation")

    # df5 = pd.read_csv('./debug_info/activationfunction/Sigmoidsaturation/result.csv')

    # analysis_show(df5, epoch)

    # st.markdown("## Rule5: Tanh Saturation")

    # df6 = pd.read_csv('./debug_info/activationfunction/Tanhsaturation/result.csv')

    # analysis_show(df6, epoch)
