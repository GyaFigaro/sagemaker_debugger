import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import matplotlib.pyplot as plt

df_data = pd.read_csv('./debug_info/weights/UpdateTooSmall.csv')
df_result = pd.read_csv('./debug_info/weights/UpdateResult.csv')


st.write('数据读取完成')

update_result = df_result.to_numpy()
Small = False
if update_result[0,1]=='None':
    Small = False
else:
    Small = True
if Small==False:
    st.title('该模型无更新过小情况')
if Small==True:
    st.title('该模型出现更新过小情况')
    small_step = update_result.shape[0]
    st.write('step      update_ratio')
    for i in range(small_step):
        st.write(update_result[i,1],'   ',update_result[i,2])

check_data = st.checkbox('查看数据')
if check_data:
    st.dataframe(df_data)
check_image = st.checkbox('显示图片')
if check_image:
    update_data = df_data.to_numpy()
    steps = update_data.shape[0]
    list_ratio = []
    x = []
    for i in range(steps):
        x.append(i+1)
        list_ratio.append(update_data[i,1])
    fig, ax = plt.subplots()
    ax.plot(x,list_ratio,linewidth=1, color="orange", marker="o",label="variance ratio")
    st.pyplot(fig)