# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 13:56:58 2025

@author: quich
"""

import joblib
import numpy as np

Predicitor = joblib.load('Inverter_Maintainance_Steps_Model.joblib')

Test = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

Prediction = Predicitor.predict(Test)
print("Maintainance Steps",Prediction)