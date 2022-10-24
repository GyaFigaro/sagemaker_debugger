import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Rule: 过拟合")
st.sidebar.header("Data Demo")
f = open('./debug_info/result4.txt','r')
result = f.read()
f.close()
if result=="True":
    st.write(
        """无过拟合"""
    )
else:
    st.write(
        """过拟合"""
    )
DATE_COLUMN = 'date/time'
def load_data(nrows):
    data = pd.read_csv('./debug_info/data4.csv')
    return data

data = load_data(3)

st.line_chart(data,x="steps")

expander = st.expander("See explanation")
if result=="True":
    expander.write("""
        无过拟合，请继续您的工作。
    """)
else:
    expander.write("""
        建议调整模型。
    """)