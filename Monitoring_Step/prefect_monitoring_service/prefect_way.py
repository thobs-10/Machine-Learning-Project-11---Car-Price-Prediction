# import dependencies
import json
import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

import pyarrow.parquet as pq
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    DataDriftProfileSection, RegressionPerformanceProfileSection)

from prefect import flow, task

from pymongo import MongoClient
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection


MONGO_CLIENT_ADDRESS = "mongodb://localhost:27017/"
MONGO_DATABASE = "prediction_service"
PREDICTION_COLLECTION = "data"
REPORT_COLLECTION = "report"
REFERENCE_DATA_FILE = "../data/green_tripdata_2021-03.parquet" # change the data to be the one needed here
TARGET_DATA_FILE = "target.csv" # look for the correct file to place here
MODEL_FILE = os.getenv('MODEL_FILE', '../prediction_service/model.pkl') 
SCALER_FILE = os.getenv('MODEL_FILE', '../prediction_service/standard_scaler.pkl') 

# create a task for prefect, a function to upload the target variable
@task
def upload_target(filename):
    # declare the mongo client by accessing the mongo client address
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    # get  the dataase and assigining the collection a collection table name
    collection = client.get_database(MONGO_DATABASE).get_collection(PREDICTION_COLLECTION)
    # open the uploaded file
    with open(filename) as f_target:
        # read each line, for each line
        for line in f_target.readlines():
            # split the row since it is a csv file
            row = line.split(",")
            # access the id and the target, place them in the collection table called data
            collection.update_one({"id": row[0]},
                                  {"$set": {"target": float(row[1])}}
                                 )
        

# task to load reference data
@task
def load_reference_data(filename):
    # load the trained and tested model, with its pickle file
    with open(MODEL_FILE, 'rb') as f_in:
        model = pickle.load(f_in)
    with open(SCALER_FILE, 'rb') as f_in:
        scaler = pickle.load(f_in)
    
    # read the ref data from the filename, coinvert it to pandas and sample 5000 rows
    reference_data = pd.read_csv(filename).to_pandas().sample(n=1000,random_state=42) #Monitoring for 1st 1000 records
    # Create features

    features = {}
     # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
    # and place that in label encoder race as a dictionary
    features.append(reference_data['Mileage'])

    label_encoding_name = {value: key for key, value in enumerate(reference_data['name'].unique())}
    features.append(reference_data['name'].map(label_encoding_name))

    label_encoding_company = {value: key for key, value in enumerate(reference_data['company'].unique())}
    features.append(reference_data['company'].map(label_encoding_company))  
    
    features.append(reference_data['year'])

    feature_df = pd.DataFrame(data=features)
    feature_df['target'] = reference_data.price
   
    
    # create a list of teh features that will be used or are of interest
    features_x = ['Mileage', 'name', 'company', 'year']
    # transform the features usingg the dict vectorizer
    x_pred = scaler.transform(reference_data[features_x].to_dict(orient='records'))
    # place the rpediction in a columnn of the reference data
    reference_data['prediction'] = model.predict(x_pred)
    return reference_data

# function to fetch the data
@task
def fetch_data():
    '''create a dataframe of the data that i stored in mongo db'''
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    data = client.get_database(MONGO_DATABASE).get_collection(PREDICTION_COLLECTION).find()
    df = pd.DataFrame(list(data))
    return df

@task
def run_evidently(ref_data, data):
    
    # create a profile for the ride prediction data
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])

    # create a mapping for thrr features that are main focal point
    mapping = ColumnMapping(prediction="prediction", numerical_features=['Mileage','year'],
                            categorical_features=['name', 'company'],
                            datetime_features=[])
    # map the ref data, data and the mapping of features to calcutae the necessary  metrics
    profile.calculate(ref_data, data, mapping)
    # declare a dashboard, a regression performance tab
    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    # calculate the metrics in the dashboard
    dashboard.calculate(ref_data, data, mapping)
    # return the profile json and the dashboard
    return json.loads(profile.json()), dashboard

@task
def save_report(result):
    """Save evidendtly profile for ride prediction to mongo server"""
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    collection = client.get_database(MONGO_DATABASE).get_collection(REPORT_COLLECTION)
    collection.insert_one(result)

@task
def save_html_report(result, filename_suffix=None):
    """Create evidently html report file for ride prediction"""
    
    if filename_suffix is None:
        filename_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    result.save(f"car_prediction_drift_report_{filename_suffix}.html")


@flow
def batch_analyze():
    # upload the  taget csv file 
    upload_target(TARGET_DATA_FILE)
    # get the ref data
    ref_data = load_reference_data(REFERENCE_DATA_FILE)
    # fetch the data
    data = fetch_data()
    # getr teh evidently profile and dashboard
    profile, dashboard = run_evidently(ref_data, data).result()
    # save the report
    save_report(profile)
    save_html_report(dashboard)

batch_analyze()

