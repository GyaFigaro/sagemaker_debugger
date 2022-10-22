import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

show_steps = 5

st.set_page_config(page_title="ML Debug: Tensor", page_icon="📈")

st.markdown("# Tensor Rules")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.markdown("## Rule: Dying Relu")

df2 = pd.read_csv('./debug_info/activationfunction/Dyingrelu/result.csv')

st.dataframe(df2)

st.markdown("## Rule: Sigmoid Saturation")

df3 = pd.read_csv('./debug_info/activationfunction/Sigmoidsaturation/result.csv')

st.dataframe(df3)

st.markdown("## Rule: Tanh Saturation")

df4 = pd.read_csv('./debug_info/activationfunction/Tanhsaturation/result.csv')

st.dataframe(df4)