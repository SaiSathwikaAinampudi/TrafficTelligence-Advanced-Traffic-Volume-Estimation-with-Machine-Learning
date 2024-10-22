import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing

app = Flask(__name__)
model = pickle.load(open(r'model.pkl','rb'))
scale = pickle.load(open(r'scale.pkl','rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/predict',methods=["POST","GET"])# route to show the show predictions in a web UI
def predict():
    # rendering the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    features_values= [np.array(input_feature)]  
    names = [['holiday','temp','rain','snow','weather','year','month','day','hours','minutes','seconds']]
    data = pandas.DataFrame(features_values,columns=names)

    data_scaled = scale.transform(data)
    data_scaled = pandas.DataFrame(data_scaled, columns = names)
    #predictions using the loaded model file
    prediction=model.predict(data_scaled)
    print(prediction)
    text = "The Estimated Traffic Volume is :"
    return render_template("index.html",prediction_text = text + str(prediction))
    #showing the predication results in a UI
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port, debug=True,use_reloader=False)
