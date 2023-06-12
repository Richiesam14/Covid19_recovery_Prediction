import json                                   ## Tested with cust input data, everything seems working incl API; 
import pickle                                 ## Verdict : pass to F'End 
import joblib

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = joblib.load(open('covpred_model.pkl', 'rb'))
scalar = pickle.load(open('scalingg.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  ## UI

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # Convert data values to floats,coz of our d.type being int64 for col    
    float_data = {key: float(value) for key, value in data.items()}     ## looping dict and make val to float
    new_data = scalar.transform(np.array(list(float_data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(float(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

