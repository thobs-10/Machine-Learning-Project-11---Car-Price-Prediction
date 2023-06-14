import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
from typing import cast
import mlflow
import requests
import json
from typing import Tuple
import logging


# read the saved pickle packages
model = pickle.load(open('dataset/pipe.pkl','rb'))
scaler = pickle.load(open('dataset/standard_scaler.pkl','rb'))
df = pd.read_csc('dataset/preprocessed_dataset.csv')

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
    pass