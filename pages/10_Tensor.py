import streamlit as st
import time
import numpy as np
import os
import pandas as pd
import json

show_steps = 5

os

st.set_page_config(page_title="ML Debug: Tensor", page_icon="📈")

st.markdown("# Tensor Rules")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.markdown("## Rule: All Values Zero")




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

df1 = pd.read_csv('./debug_info/tensor/AllZeroValues/result.csv')

st.dataframe(df1)

# i = show_steps
# while i >= 0:
#     for step in df1.columns():

st.markdown("## Rule: Tensor Not Changed")

df3 = pd.read_csv('./debug_info/tensor/ValuesUnchanged/result.csv')

st.dataframe(df3)

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
