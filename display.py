from unittest import result
import streamlit as st
import pandas as pd
import altair as alt
import os
from urllib.error import URLError
import json

@st.cache
def show_epoch(epoch):
    st.write("epoch: ", epoch)

@st.cache
def analysis_show(df, epoch):
    epoch_result = {"result":'True', 'step':0}
    show_epoch(epoch)
    st.write(df)

st.set_page_config(page_title="ML Debug Demo", page_icon="📊")

st.markdown("# ML Debug Demo")

# st.sidebar.success("Select a type of rule above.")

with open("./debug_info/epoch_info.json",'r') as load_f:
    load_epo = json.load(load_f)
    epoch = load_epo['epoch_num']

    st.markdown("# Tensor Rules")

    st.markdown("## Rule: All Values Zero")

    df1 = pd.read_csv('./debug_info/tensor/AllZeroValues/result.csv')
    
    analysis_show(df1, epoch)
    

    # df1 = pd.read_csv('./debug_info/tensor/data.csv')

    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)

    # for i in range(1, 101):
    #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows(new_rows)
    #     progress_bar.progress(i)
    #     last_rows = new_rows
    #     time.sleep(0.05)



    st.markdown("## Rule: Tensor Not Changed")

    df3 = pd.read_csv('./debug_info/tensor/ValuesUnchanged/result.csv')

    analysis_show(df3, epoch)

    st.markdown("# Tensor Rules")

    st.write(
        """This demo illustrates a combination of plotting and animation with
    Streamlit. We're generating a bunch of random numbers in a loop for around
    5 seconds. Enjoy!"""
    )

    st.markdown("## Rule: Dying Relu")

    df4 = pd.read_csv('./debug_info/activationfunction/Dyingrelu/result.csv')

    analysis_show(df4, epoch)

    st.markdown("## Rule: Sigmoid Saturation")

    df5 = pd.read_csv('./debug_info/activationfunction/Sigmoidsaturation/result.csv')

    analysis_show(df5, epoch)

    st.markdown("## Rule: Tanh Saturation")

    df6 = pd.read_csv('./debug_info/activationfunction/Tanhsaturation/result.csv')

    analysis_show(df6, epoch)

    # def load_data(nrows):
    #     data = pd.read_csv('./pages/data.csv')
    #     return data


    # data = load_data(3)


    # st.bar_chart(data, x='labels', y='count')