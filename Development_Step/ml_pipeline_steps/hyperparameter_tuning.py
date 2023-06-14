import argparse
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# running sqlite for backend storage
# mlflow ui --backend-store-uri sqlite:///mlflow.db( you can customize the name of the db to the one you prefer)
mlflow.set_tracking_uri("http://127.0.0.1:5000") # the url to visualize the experiments to the local host tracking server
# keep in mind this url for the server is still running and the server is running also in the backend store
mlflow.set_experiment("car-price-prediction-tuning-1") # experiment name

# the run function  is responsible for beginning the experiment run
def run(data_path, num_trials):
    # load dataset for train and validation
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

    def objective(params):
        # an inner function to track the runs and record the parameters
        with mlflow.start_run():
            # log the parameters that are past in arguments
            mlflow.log_params(params)
            mlflow.set_tag("model-name","random-forest-regressor")
            # instantiate the model and fit the model
            rfr = RandomForestRegressor(**params)
            rfr.fit(X_train, y_train)
            y_pred = rfr.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
    # define the search space
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }   
    # use the minimizing function of huper opt to minimize the objective function
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

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
        default=20,
        help="the number of parameter evaluations for the optimizer to explore."
    )
    args = parser.parse_args()

    run(args.data_path, args.max_evals)


