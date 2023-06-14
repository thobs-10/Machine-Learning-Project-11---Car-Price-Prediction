import os
import json
import boto3
import base64
import pandas as pd
import numpy as np

import mlflow

kinesis_client = boto3.client('kinesis')

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'car_price_predictions')


RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
# logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


# load scaler
logged_scaler = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
# logged_model = f'runs:/{RUN_ID}/model'
sc = mlflow.pyfunc.load_model(logged_scaler)


TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

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

def load_scaler(RUN_ID):

    logged_scaler = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
    # logged_model = f'runs:/{RUN_ID}/model'
    scaler = mlflow.pyfunc.load_model(logged_scaler)
    return scaler

def predict(features):
    #sc = load_scaler('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 11 - Car Price Prediction/dataset/standard_scaler.pkl')
    #sc = load_scaler(RUN_ID=RUN_ID)
    x = np.array(features).reshape(1, -1)
    scaled_features = sc.transform(x)
    preds = model.predict(scaled_features)
    return float(preds[0])


def lambda_handler(event, context):
    # print(json.dumps(event))
    
    predictions_events = []
    
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)

        # print(ride_event)
        ride = ride_event['car']
        ride_id = ride_event['car_id']
    
        features = prepare_features(ride)
        prediction = predict(features)
    
        prediction_event = {
            'model': 'car_price_prediction_model',
            'version': '123',
            'prediction': {
                'price_prediction': prediction,
                'ride_id': ride_id   
            }
        }

        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )
        
        predictions_events.append(prediction_event)


    return {
        'predictions': predictions_events
    }


