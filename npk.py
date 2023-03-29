# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:15:32 2023

@author: PRAMILA
"""

import numpy as np 
import pandas as pd 
import streamlit as st
st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>VEG ANALYST</h1>", unsafe_allow_html=True)
st.write("hi")
import pickle
global df
labels=['Beans', 'okra', 'onion', 'potato', 'tomato', 'watermelon']
fertilizer_labels=['Ammonium chloride',
 'Ammonium phosphate',
 'Ammonium sulphate',
 'Anhydrous ammonia',
 'Calcium ammonium nitrate',
 'Diammonium phosphate (DAP)',
 'Double super phosphate',
 'Muriate of potash (MOP)',
 'Potassium magnesium sulphate',
 'Potassium nitrate',
 'Potassium polyphosphate',
 'Rock phosphate',
 'Single super phosphate',
 'Sulphate of potash',
 'Triple super phosphate',
 'Urea']
n = st.slider('Nitrogen', min_value=0, step=1, max_value=500,value=500)
p = st.slider('Phosphorous', min_value=0, step=1, max_value=500,value=500)
k = st.slider('Potassium', min_value=0, step=1, max_value=500,value=500)
test_data=[[n,p,k]]

if st.button("Predict"):  
 
    random_forest_model = pickle.load(open('randomforest_model.pkl','rb'))
    o=random_forest_model.predict(test_data)
    print(o)
    print(labels[int(o[0])])
    
    s="Randomforest PREDICTS : "+labels[int(o[0])]
    st.info(s)
    
    sgd_classifier = pickle.load(open('sgd_model.pkl','rb'))
    results = pd.DataFrame(sgd_classifier.predict_proba(test_data).transpose())
    results.rename(columns={0: "score"}, inplace=True)
    
    results.reset_index(level=0, inplace=True)
    results.sort_values(by="score", ascending=False, inplace=True, ignore_index=True)
    #st.write(results)
    total=results['score'].isnull().sum()
    if total==len(results):
        st.success("no fertilizer needed")
    else:
       st.error("The following fertilizers are needed")
       for i in range(len(results)):
            if results.loc[i, "score"] > 0.001:
                st.warning(str(fertilizer_labels[results.loc[i, "index"]]) + " (" + str(round(100 * results.loc[i, "score"])) + "%)") 
        
