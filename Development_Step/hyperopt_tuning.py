import argparse
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_squared_error


mlflow.xgboost.autolog(disable=True)

# mlflow ui --backend-store-uri sqlite:///mlflow.db( you can customize the name of the db to the one you prefer)
mlflow.set_tracking_uri("http://127.0.0.1:5000") # the url to visualize the experiments to the local host tracking server
# keep in mind this url for the server is still running and the server is running also in the backend store
mlflow.set_experiment("car-price-prediction-hyperopt-1") # experiment name

def run(data_path, num_trials):

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
    X_valid = np.asarray(X_test_df)
    y_train = np.asarray(y_train_df)
    y_valid = np.asarray(y_test_df)

    def objective():

        with mlflow.start_run():
            
            # the best parameters that minimize the rsme and optimize the objective function, taken from mlflow
            best_params = {
                'max_depth':5 ,
                'min_samples_leaf':	4,
                'min_samples_split':8,
                'n_estimators':28,
                'random_state':42
 
            }
            # log the parameters on mlflow
            mlflow.log_params(best_params)
            # now train the model with the best parames chosen from the run that performed better thann others
            random_forest_reg = rfr(max_depth=5,
                                    min_samples_leaf=4,
                                    min_samples_split=8,
                                    n_estimators=28,
                                    random_state=42)
            random_forest_reg.fit(X=X_train, y=y_train)

            y_pred = random_forest_reg.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            # logging it into the mlflow
            #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
            # logging the model artifact
            mlflow.sklearn.log_model(random_forest_reg, artifact_path="models_mlflow")

    objective()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="the number of parameter evaluations for the optimizer to explore."
    )
    args = parser.parse_args()

    run(args.data_path, args.max_evals)