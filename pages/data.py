import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Data Demo")
st.sidebar.header("Data Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)
DATE_COLUMN = 'date/time'


def load_data(nrows):
    data = pd.read_csv('./pages/data.csv')
    return data


data = load_data(3)


st.bar_chart(data, x='labels', y='count')