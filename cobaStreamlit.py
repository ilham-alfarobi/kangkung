#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np


# In[2]:


st.write("Sonic Bloom merupakan pancaran frekuensi tinggi seperti musik adalah salah satu contoh frekuensi tinggi,yang digunakan untuk memperlebar stomata pada daun. Keterangan Data Yang Digunakan \n")
st.write("1. Data tingi: Merupakan Tinggi dari Tumbuhan kangkung hidroponik pada saat akan diprediksi \n")
st.write("2. Suhu air: Merupakan suhu air pada waktu dilakukan prediksi(sekarang)  \n")
st.write("3. Suhu ruang: Merupakan suhu ruang pada waktu dilakukan prediksi(sekarang) \n")
st.write("4. Jenis Musik: Merupakan jenis musik yang digunakan untuk perlakuan Sonic Bloom terhadap tumbuhan kangkung hidroponik")


# In[4]:


st.write("Overview Data")
myData = pd.read_csv('dataset_sonicbloom.csv')
st.dataframe(myData)

st.write("Deskripsi Data")

st.dataframe(myData.describe())
# Preproccessing Data
st.write("Dilakukan Preprocessing Data dimana Fitur dan Labelnya akan Dipisah")
# Memisahkan Label Dan Fitur 
X = myData[['tinggi','suhu_air','suhu_ruang','jenis_musik']]
y = myData['hari_real']
st.write("## Input Data X",X)
st.write("## Label Data y",y)


# In[5]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[6]:


from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators=50, criterion='entropy', min_samples_split= 6, min_samples_leaf= 1, max_depth= 40, random_state=42)
model = classifier_forest.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Dengan Menggunakan Random Forest Diperoleh Skor Untuk Data Test")
st.write(f'On Train Accuracy - : {model.score(X_train,y_train):.3f}')
st.write(f'On Test Accuracy - : {model.score(X_test,y_test):.3f}')


# In[7]:


st.write("# Sekarang Silahkan Masukan nilai berikut Untuk Mengetahui Prediksi umur kangung hidroponik anda")

form = st.form(key='my-form')
inputTinggi = form.number_input("Masukan tinggi tumbuhan: ", 0)
inputSuhuair = form.number_input("Masukan suhu air sekarang ini: ", 0)
inputSuhuruang = form.number_input("Masukan suhu ruang sekarang ini: ", 0)
inputjenismusik = form.number_input("Masukan jenis musik yang digunakan \n1.Dangdut \n2.Jazz \n3.Murottal\n4.Tanpa musik: ", 0)
submit = form.form_submit_button('Submit')

completeData = np.array([inputTinggi, inputSuhuair, inputSuhuruang, 
                        inputjenismusik]).reshape(1, -1)
prediction = model.predict(completeData)
st.write('Tekan Submit Untuk Melihat Prediksi umur tumbuhan anda')


# In[ ]:




