import argparse
import os
import pickle
import pandas as pd
import  numpy as np

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "car-price-prediction-tuning-1"
EXPERIMENT_NAME = "random-forest-model_registry"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}

def train_and_log_model(data_path, params):
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

    with mlflow.start_run():
        best_params = {
                'max_depth':5 ,
                'min_samples_leaf':	4,
                'min_samples_split':8,
                'n_estimators':28,
                'random_state':42
 
            }
        params = space_eval(SPACE, params)
        rf = RandomForestRegressor(max_depth=5,
                                    min_samples_leaf=4,
                                    min_samples_split=8,
                                    n_estimators=28,
                                    random_state=42)
        rf.fit(X_train, y_train)

        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        train_rmse = mean_squared_error(y_train, rf.predict(X_train), squared=False)
        mlflow.log_metric("train_rmse", train_rmse)


def run(data_path, log_top):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # register the best model
    model_uri = f"runs:/{best_run.info.run_id }/model"
    mlflow.register_model(
        model_uri=model_uri, 
        name="price-pred"
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
