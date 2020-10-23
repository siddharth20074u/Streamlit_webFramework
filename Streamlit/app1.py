#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:21:54 2020

@author: siddharthsmac
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'Hello Siddharth'

def predict_note_authentication(variance, skewness, curtosis, entropy):
       
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

def main():
    st.title('Bank Authenticator')
    html_temp = """
    <div style = 'background-color:tomato; padding: 10px'>
    <h2 style = 'color:white; text-align:center;'> Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    variance = st.text_input('variance', 'Type Here')
    skewness = st.text_input('skewness', 'Type Here')
    curtosis = st.text_input('curtosis', 'Type Here')
    entropy = st.text_input('entropy', 'Type Here')
    result = ""
    
    if st.button('Predict'):
        result = predict_note_authentication(variance, skewness, curtosis, entropy)
    st.success('The output is {}'.format(result))
    
    if st.button('About'):
        st.text('Lets Learn')
        st.text('Built with StreamLit')
        st.text('Developed by Siddharth Shankar')



if __name__ =='__main__':
    main()