import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Data Demo", page_icon="📊")

st.markdown("# Rule: 欠拟合")

f = open('./debug_info/result6.txt','r')
result = f.read()
f.close()
if result=="True":
    st.write(
        """无欠拟合"""
    )
else:
    st.write(
        """欠拟合"""
    )
DATE_COLUMN = 'date/time'

data1 = pd.read_csv('./debug_info/data6.csv')

st.line_chart(data1,x="steps")

expander = st.expander("See explanation")
if result=="True":
    expander.write("""
        无欠拟合，请继续您的工作。
    """)
else:
    expander.write("""
        建议调整模型。
    """)