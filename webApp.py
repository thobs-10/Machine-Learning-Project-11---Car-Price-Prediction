import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
from typing import cast
import json
from typing import Tuple
import logging


# read the saved pickle packages
model = pickle.load(open('model.pkl','rb'))
#scaler = pickle.load(open('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\standard_scaler.pkl','rb'))
df = pd.read_csv('dataset/preprocessed_dataset.csv')

st.title("Car Price Prediction")

#Mileage
Mileage = st.number_input('Mileage')

#company name
company_name = st.selectbox('Company Name', df['company'].unique())

# model name
model_name = st.selectbox('Model Name', df['name'].unique())

#Year
year = st.number_input('year')

if st.button('Predict Price'):

    input_tuple = {"Mileage": [Mileage],
    "company": [company_name],
    "name": [model_name],
    "year": [year]}

    input_df = pd.DataFrame(data=input_tuple)

    features = []
    # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
    # and place that in label encoder race as a dictionary
    features.append(input_df['Mileage'])

    label_encoding_name = {value: key for key, value in enumerate(input_df['name'].unique())}
    features.append(input_df['name'].map(label_encoding_name))

    label_encoding_company = {value: key for key, value in enumerate(input_df['company'].unique())}
    features.append(input_df['company'].map(label_encoding_company))  
    
    features.append(input_df['year'])

    x = np.array(features).reshape(1, -1)

    y_pred = model.predict(x)

    # st.title(y_pred)
    st.title("The predicted price of this car configuration is " + str(round((y_pred[0]),2)))