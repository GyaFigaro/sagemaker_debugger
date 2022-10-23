import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Classifier Confusion")
st.sidebar.header("Data Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)
DATE_COLUMN = 'date/time'

def load_data(nrows):
    data = pd.read_csv('./debug_info/data5.csv')
    return data

data = load_data(3)
st.dataframe(data)

expander = st.expander("./debug_info/See explanation")
expander.write("""
    The chart above shows some numbers I picked for you.
    I rolled actual dice for these, so they're *guaranteed* to
    be random.
""")