# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:05:00 2023

@author: tarun
"""

import pandas as pd
import numpy as np 
import pickle

pickled_model = pickle.load(open("C:/Users/tarun/Downloads/wine quality deploy/winequality.pkl","rb"))

input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

input_data_array = np.asarray(input_data)

input_reshaped_data = input_data_array.reshape(1,-1)

output =pickled_model.predict(input_reshaped_data)

print(output[0])