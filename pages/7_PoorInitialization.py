import streamlit as st
import pandas as pd
import numpy as np
import smdebug.pytorch as smd
import matplotlib.pyplot as plt

df_data = pd.read_csv('./debug_info/weights/PoorInitialization.csv')
df_result = pd.read_csv('./debug_info/weights/PoorResult.csv')

#my_bar = st.progress(0)
#for percent_complete in range(100):
#    time.sleep(0.1)
#    my_bar.progress(percent_complete + 1)
st.write('数据读取完成')

poor_result = df_result.to_numpy()

num_layers = poor_result.shape[0]
list_result = []
for i in range(num_layers):
    list_result.append(poor_result[i,1])
if len(list_result)==0:
    st.title('模型参数无初始化不当情况')
if len(list_result)!=0:
    st.title('该模型参数初始化不当')
    st.write('初始化不当的层为：')
    for i in list_result:
        st.write('  '+i)


check_data = st.checkbox('查看数据')
if check_data:
    st.dataframe(df_data)
check_image = st.checkbox('显示图片')
if check_image:
    poor_data = df_data.to_numpy()
    num_steps = poor_data.shape[0]
    num_sort = int((poor_data.shape[1]-1)/2)
    list_name = []
    for i in range(num_sort):
        j = 2*i+1
        list_name.append(poor_data[0,j])
    layers = st.radio(
        "Which layer of data graph do you want to view?",
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
            st.pyplot(fig)