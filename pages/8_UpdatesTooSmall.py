import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import matplotlib.pyplot as plt

df_data = pd.read_csv('./debug_info/weights/UpdateTooSmall.csv')
df_result = pd.read_csv('./debug_info/weights/UpdateResult.csv')


st.title('权重更新情况检测')
st.write('数据读取完成')

update_result = df_result.to_numpy()
Small = False
if update_result[0,1]=='None':
    Small = False
else:
    Small = True
if Small==False:
    st.write('该模型无更新过小情况')
if Small==True:
    st.write('该模型出现更新过小情况')
    small_step = update_result.shape[0]
    st.write('step      update_ratio')
    for i in range(small_step):
        st.write(update_result[i,1],'   ',update_result[i,2])

check_data = st.checkbox('查看数据')
if check_data:
    st.write("权重更新比率如下：")
    st.dataframe(df_data)
check_image = st.checkbox('查看图片')
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
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus']=False
    plt.xlabel("update step")
    plt.ylabel("weight update ratio")
    #plt.xlabel("训练步数")
    #plt.ylabel("权重更新比率")
    st.pyplot(fig)