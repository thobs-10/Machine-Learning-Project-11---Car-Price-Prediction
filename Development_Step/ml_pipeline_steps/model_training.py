# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import argparse

import mlflow # for tracking the experiment

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
#import xgboost as xgb
# lets try other classification models and see
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

# get the training and test datasets
#X_train_df = pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\X_train.csv')
#X_test_df=pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\X_test_df.csv')
#y_train_df=pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\y_train_df.csv')
#y_test_df=pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\y_test_df.csv')


#set the tracking uri - needed for the ssqlite 
mlflow.set_tracking_uri("sqlite:///car-price.db")
#set the experiment
mlflow.set_experiment("car-price-experiment-1")

#enable auto logging so that everything can be automatically loggeed to MLflow 
mlflow.sklearn.autolog()

def run(data_path):
    for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR, SVR,
        KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, XGBRegressor):

            with mlflow.start_run():
                # get the data
                X_train_df = pd.read_csv(f'{data_path}'"X_train.csv")
                X_test_df = pd.read_csv(f'{data_path}'"X_test_df.csv")
                y_train_df = pd.read_csv(f'{data_path}'"y_train_df.csv")
                y_test_df = pd.read_csv(f'{data_path}'"y_test_df.csv")

                # remove the unnecessary column
                X_train_df.drop(columns='Unnamed: 0', axis=1, inplace=True)
                y_train_df.drop(columns='Unnamed: 0', axis=1, inplace=True)
                #test set
                X_test_df.drop(columns='Unnamed: 0', axis=1, inplace=True)
                y_test_df.drop(columns='Unnamed: 0', axis=1, inplace=True)


                # change the pandas datasets to numpy arrays
                X_train = np.asarray(X_train_df)
                X_test = np.asarray(X_test_df)
                y_train = np.asarray(y_train_df)
                y_test = np.asarray(y_test_df)

                #tags of the runs
                mlflow.set_tag("data-scientist", "thobela")
                mlflow.log_param("train-x-data-path", "./dataset/X_train.csv")
                mlflow.log_param("train-y-data-path", "./dataset/y_train.csv")
                mlflow.log_param("valid-x-data-path", "./dataset/X_test.csv")
                mlflow.log_param("valid-y-data-path", "./dataset/y_test.csv")
                #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

                mlmodel = model_class()
                mlmodel.fit(X_train, y_train)

                y_pred = mlmodel.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\",
        help="the location where the processed car price data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)
