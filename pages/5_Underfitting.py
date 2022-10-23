import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Loss Not Decreasing")
st.sidebar.header("Data Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)
DATE_COLUMN = 'date/time'

def load_data(nrows):
    data = pd.read_csv('./debug_info/data3.csv')
    return data

data1 = pd.read_csv('./debug_info/data62.csv')

st.line_chart(data1,x="steps_train",y="loss_train")

data2 = pd.read_csv('./debug_info/data61.csv')

st.line_chart(data2,x="steps_test",y="loss_test")

expander = st.expander("See explanation")
expander.write("""
    The chart above shows some numbers I picked for you.
    I rolled actual dice for these, so they're *guaranteed* to
    be random.
""")