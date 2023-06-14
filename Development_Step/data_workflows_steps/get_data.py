import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import logging
#%matplotlib inline

from zenml.steps import Output, step

filename = 'C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\quikr_car.csv'

class Data_Importer():

    def __init__(self) -> None:
        pass

    def importing_data(self) -> pd.DataFrame:
         dataset = pd.read_csv(filename)
         return dataset


@step
def import_data() -> Output(
     dataset = pd.DataFrame   
):
    try:
        get_data_class = Data_Importer()
        dataset = get_data_class.importing_data()
        return dataset
    except Exception as e:
        logging.error(e)
        raise e

# send data for data integration
#orchest.output((dataset),name='imported-dataset')


