import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Rule: 损失未减小")
f = open('./debug_info/result3.txt','r')
result = f.read()
f.close()
if result=="True":
    st.write(
        """无损失不减少"""
    )
else:
    st.write(
        """存在损失不减少"""
    )
DATE_COLUMN = 'date/time'

def load_data(nrows):
    data = pd.read_csv('./debug_info/data3.csv')
    return data

data = load_data(3)

st.line_chart(data,x="steps",y="losses")

expander = st.expander("See explanation")
if result=="True":
    expander.write("""
        无损失不减少，请继续您的工作。
    """)
else:
    expander.write("""
        建议调整模型参数或减少模型训练次数。
    """)