import streamlit as st
import numpy as np
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

st.title("Car Price Prediction")



