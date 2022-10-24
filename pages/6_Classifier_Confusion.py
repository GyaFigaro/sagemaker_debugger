import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Classifier Confusion")
f = open('./debug_info/result5.txt','r')
result = f.read()
f.close()
if result=="True":
    st.write(
        """模型预测准确率良好"""
    )
else:
    st.write(
        """模型预测准确率未达标"""
    )
DATE_COLUMN = 'date/time'
def load_data(nrows):
    data = pd.read_csv('./debug_info/data5.csv')
    return data

data = load_data(3)
st.dataframe(data)

expander = st.expander("See explanation")
if result=="True":
    expander.write("""
        模型预测准确率良好。
    """)
else:
    expander.write("""
        模型预测准确率未达标。
    """)