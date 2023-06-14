import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

with open('model.pkl', 'rb') as f_in:
    (model) = pickle.load(f_in)

with open('Standard_scaler.pkl', 'rb') as f_in:
    (sc) = pickle.load(f_in)

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


def predict(features):
    #sc = load_scaler('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 11 - Car Price Prediction/dataset/standard_scaler.pkl')
    #sc = load_scaler(RUN_ID=RUN_ID)
    x = np.array(features).reshape(1, -1)
    scaled_features = sc.transform(x)
    preds = model.predict(scaled_features)
    return float(preds[0])



app = Flask('price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

