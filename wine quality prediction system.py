# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:09:46 2023

@author: tarun
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st


pickled_model = pickle.load(open("C:/Users/tarun/Downloads/wine quality deploy/winequality.pkl","rb"))


def wine_quality(input_data):
   input_data_array = np.asarray(input_data)

   input_reshaped_data = input_data_array.reshape(1,-1)

   output = pickled_model.predict(input_reshaped_data)

   return("The quality of red wine is {}".format(output[0]))

def main():
    st.title("RED WINE QUALITY PREDICTION")
    st.write("Please enter the following data")
    st.image("https://tse4.mm.bing.net/th?id=OIP.x-8U7IR06Fwcgwq6uxoTLwHaE7&pid=Api&P=0&h=180")
    
    fixed_acidity = st.text_input("Enter the value of Fixed Acidity")
    volatile_acidity = st.text_input("Enter the value of Volatile Acidity")
    citric_acid = st.text_input("Enter the value of Citric Acid")
    residual_sugar = st.text_input("Enter the value of Residual Sugar")
    chlorides = st.text_input("Enter the value of Chlorides")
    free_sulfur_dioxide = st.text_input("Enter the value of Free Sulphur Dioxide")
    total_sulfur_dioxide = st.text_input("Enter the value of Total Sulphur Dioxide")
    density = st.text_input("Enter the value of Density")
    pH = st.text_input("Enter the value of pH")
    sulphates = st.text_input("Enter the value of Sulphates")
    alcohol = st.text_input("Enter the value of Alcohol")
    
    quality = ''
    
    
    if st.button("Check Wine Quality"):
        quality = wine_quality([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        st.balloons()
        
    st.success(quality)
    st.image("http://servingjoy.com/wp-content/uploads/2014/12/Bottle-and-glass-of-red-wine.jpg")
    
if __name__ == "__main__":
    main()