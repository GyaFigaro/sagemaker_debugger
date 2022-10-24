import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import matplotlib.pyplot as plt
import os

flag = 1
df_data = pd.read_csv('./debug_info/weights/PoorInitialization.csv')

if os.path.exists('./debug_info/weights/PoorResult.csv'):
    df_result = pd.read_csv('./debug_info/weights/PoorResult.csv')
else:
    flag = 0

st.title('权重初始化情况检测')
st.write('数据读取完成')

if flag == 0:
    st.write('模型参数无初始化不当情况')
else:
    poor_result = df_result.to_numpy()
    num_layers = poor_result.shape[0]
    list_result = []
    for i in range(num_layers):
        list_result.append(poor_result[i,1])
    poor_result = df_result.to_numpy()
    st.write('该模型参数初始化不当')
    st.write('初始化不当的层为：')
    for i in list_result:
        st.write('  '+i)


check_data = st.checkbox('查看数据')
if check_data:
    st.write("各层权重方差如下：")
    st.dataframe(df_data)
check_image = st.checkbox('查看图片')
if check_image:
    poor_data = df_data.to_numpy()
    num_steps = poor_data.shape[0]
    num_sort = int((poor_data.shape[1]-1)/2)
    list_name = []
    for i in range(num_sort):
        j = 2*i+1
        list_name.append(poor_data[0,j])
    layers = st.radio(
        "您想查看哪一层的权重的方差值?",
        list_name)
    for i in range(num_sort):
        if layers==list_name[i]:
            k = 2*i+2
            layer_data = []
            x = []
            for j in range(num_steps):
                x.append(j)
                layer_data.append(poor_data[j,k])
            fig, ax = plt.subplots()
            ax.plot(x,layer_data,linewidth=1, color="orange", marker="o",label="variance ratio")
            plt.xlabel("update step")
            plt.ylabel("weight variance")
            st.pyplot(fig)