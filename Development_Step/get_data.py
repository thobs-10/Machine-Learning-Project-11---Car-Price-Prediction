import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline

import orchest

filename = 'C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\quikr_car.csv'

# import the dataset
dataset = pd.read_csv(filename)

# send data for data integration
orchest.output((dataset),name='imported-dataset')


