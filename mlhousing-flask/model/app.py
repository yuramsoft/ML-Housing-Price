# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 10:50:38 2022

@author: IBRAHIM MUSTAPHA
"""

#import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import pickle
import os


app = Flask(__name__)

model = pickle.load(open('estate_forest.pkl', 'rb'))  # loading the model

@app.route("/")
def home():
    return render_template('index.html')
   # return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='House price is : ${}'.format(output))

#     return render_template('predict.html',)

if __name__ == "__main__":
   app.run(host='0.0.0.0',port=5000, debug=os.environ.get('DEBUG')=='1')
   
 