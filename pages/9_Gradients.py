import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import matplotlib.pyplot as plt
import seaborn as sns

df_data = pd.read_csv('./debug_info/gradients/gradients.csv')
df_result = pd.read_csv('./debug_info/gradients/GradientsResult.csv')

st.write('数据读取完成')

g_result = df_result.to_numpy()
if  g_result[0,2]==True:
    st.write('该模型出现了梯度消失现象')
if  g_result[0,2]==False:
    st.write('该模型未出现梯度消失现象')
if g_result[1,2]==True:
    st.write('该模型出现了梯度爆炸现象')
if g_result[1,2]==False:
    st.write('该模型未出现梯度爆炸现象')

check_data = st.checkbox('查看数据')
if check_data:
    st.dataframe(df_data)
check_image = st.checkbox('显示图片')
if check_image:
    g_data = df_data.to_numpy()
    num_steps = g_data.shape[0]
    num_layers = g_data.shape[1]-1
    for i in range(num_steps):
        if i==0:
            new_array = g_data[0,1: ]
        if i!=0:
            i_array = g_data[i,1: ]
            new_array = np.concatenate((new_array,i_array))
    new_array = new_array.reshape(num_steps,num_layers)
    sns.set_theme()
    ax = sns.heatmap(new_array,annot=True,fmt=".4f",linewidths=.9000)
    plt.savefig("./debug_info/gradients/ThermalMap")
    st.write('热力图')
    st.image("./debug_info/gradients/ThermalMap.png")
    #layers = st.radio(
        #"Which layer of data graph do you want to view?",
        #list_name)
    #for i in range(num_sort):
        #if layers==list_name[i]:
            #k = 2*i+2
            #layer_data = []
            #x = []
            #for j in range(num_steps):
                #x.append(j)
                #layer_data.append(poor_data[j,k])
            #fig, ax = plt.subplots()
            #ax.plot(x,layer_data,linewidth=1, color="orange", marker="o",label="variance ratio")
            #st.pyplot(fig)