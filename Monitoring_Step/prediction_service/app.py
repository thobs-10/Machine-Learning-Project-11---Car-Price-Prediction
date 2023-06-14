import os
import pickle

import pandas as pd
import numpy as np

import requests
from flask import Flask
from flask import request
from flask import jsonify

# for the mangoDb database
from pymongo import MongoClient


# to access the predictive model that has the dictvevtorizer and moddel
MODEL_FILE = os.getenv('MODEL_FILE', 'model.pkl')
# access the scaler
SCALER_FILE = os.getenv('SCALER_FILE','standard_scaler.pkl')
# the address where the data will be sent for monitoring using evidently 
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
# MANGODB for storing the data before it goes to the premotheous and evidently or monitoring service
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
# openning the linear reg model file and dict vectorizer
with open(MODEL_FILE, 'rb') as f_in:
    model = pickle.load(f_in)

with open(SCALER_FILE, 'rb') as f_in:
    scaler = pickle.load(f_in)


# creating a flask
app = Flask('price-prediction')
# creating the database
mongo_client = MongoClient(MONGODB_ADDRESS)
# getting the database and calling the predictive service
db = mongo_client.get_database("prediction_service")
# getting the data that will be stored and sent to evidently
collection = db.get_collection("data")


@app.route('/predict', methods=['POST'])
def predict():
    # getting the json file that has the records
    record = request.get_json()

    features = prepare_features(record)

    x = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(x)
    y_pred = model.predict(scaled_features)
    # results stored in a form of a dictionary
    result = {
        'duration': float(y_pred[0]),
    }
    # create a function that will save the data predicted to the database
    save_to_db(record, float(y_pred[0]))
    # a function to send the data to evidently service
    send_to_evidently_service(record, float(y_pred[0]))
    # return tthe results in A jsonify file
    return jsonify(result)

def prepare_features(car):
    "feature engineer and transform the incoming request"
    df = pd.DataFrame(data=car)
    features = {}
     # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
    # and place that in label encoder race as a dictionary
    features.append(df['Mileage'])

    label_encoding_name = {value: key for key, value in enumerate(df['name'].unique())}
    features.append(df['name'].map(label_encoding_name))

    label_encoding_company = {value: key for key, value in enumerate(df['company'].unique())}
    features.append(df['company'].map(label_encoding_company))  
    
    features.append(df['year'])

    return features

# function that takse in the records and teh predicted results, store them in db
def save_to_db(record, prediction):
    # create a copy of the record
    rec = record.copy()
    #in the records dataframe,create a column  for the predicted values
    rec['prediction'] = prediction
    # add the recods to database
    collection.insert_one(rec)

# function to get the records and the predicted records to the monitoring service
def send_to_evidently_service(record, prediction):
    # make a copy of the records
    rec = record.copy()
    # create a column of the predicted values
    rec['prediction'] = prediction
    # qwe neeed ri post the records in a json file to the evidently service address
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/car", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
