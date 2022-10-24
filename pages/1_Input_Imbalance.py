import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Input Imbalance")
st.sidebar.header("Data Demo")

data = pd.read_csv('./debug_info/data.csv')

f = open('./debug_info/result1.txt','r')
result = f.read()
f.close()
if result=="True":
    st.write(
        """无数据集不平衡"""
    )
else:
    st.write(
        """存在数据集不平衡"""
    )
DATE_COLUMN = 'date/time'
st.bar_chart(data,x='labels',y='count')

expander = st.expander("See explanation")
if result=="True":
    expander.write("""
        无数据集不平衡，请继续您的工作。
    """)
else:
    expander.write("""
        建议根据图表增添数据过少的类的数据。
    """)