import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Rule: 数据未归一化")
f = open('./debug_info/result2.txt','r')
result = f.read()
f.close()
if result=="0":
    st.write(
        """无数据未均一化"""
    )
elif result=='1':
    st.write(
        """存在数据未均一化"""
    )
else:
    st.write(
        """检测数据量过少！"""
    )
DATE_COLUMN = 'date/time'

def load_data(nrows):
    data = pd.read_csv('./debug_info/data2.csv')
    return data

data = load_data(3)

st.line_chart(data,x='quants',y='means')

expander = st.expander("See explanation")
if result=="0":
    expander.write(
        """无数据未均一化，请继续您的工作。"""
    )
elif result=='1':
    expander.write(
        """存在数据未均一化，请调整你的数据再进行下一步。"""
    )
else:
    expander.write(
        """检测数据量过少！请增添你的数据量并再次检测。"""
    )