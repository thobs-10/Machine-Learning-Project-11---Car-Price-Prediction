import os
import uuid # generates globally unique id's numbers
import pickle
import sys

import pandas as pd
import numpy as np

import mlflow

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context


RUN_ID = "e4761ba7b84842c9b8d224b9ab3aa234"

output_file = f'output/bacth-results.parquet'
df = pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv')

def generate_uuids(n):
    car_ids = []
    for i in range(n):
        car_ids.append(str(uuid.uuid4()))
    return car_ids

df['Mileage'] = df['kms_driven'].str.split(' ').str[0]
df.drop(columns='kms_driven',axis=1, inplace=True)

@task
def read_dataset(filename:str):
    df = pd.read_csv(filename)
    # remove the unnamed column
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # before making any changes lets copy the  data and work with the copy
    df_copy = df.copy()

    # change the ask for price text value in price column and replace it with 0
    #df_copy['Price'] = df_copy['Price'].replace('Ask For Price', 0)
    return df_copy

def fix_price_datatype(df_copy):
    for i in range(len(df_copy)):
        cur_price = df_copy.loc[i,'Price']
        if(cur_price!= 0):
            cur_price = float(cur_price.replace(',',''))
            df_copy.loc[i,'Price']=cur_price

# fixing the mileage column
def fix_mileage_datatype(df_copy):
    for i in range(len(df_copy)):
        cur_price = df_copy.loc[i,'Mileage']
        if(cur_price!=0):
            cur_price = float(cur_price.replace(',',''))
            df_copy.loc[i,'Mileage']=cur_price
    return df_copy

def fix_datatypes(df_copy):
    # make the price column to be aa float
    df_copy['Price'] = df_copy['Price'].astype(float)
    df_copy['Mileage'] = df_copy['Mileage'].astype(float)
    return df_copy

# deal with the NAN
def fixing_nans(df_copy):
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    imputer.fit(df_copy.iloc[:, 3:4])
    df_copy.iloc[:, 3:4] = imputer.transform(df_copy.iloc[:, 3:4])
    return df_copy

# removee the outliers
def remove_year_outliers(df):
    # calculate the Quantiles(Q1 and Q3)
    Q1 = df['year'].quantile(0.25)
    Q3 = df['year'].quantile(0.75)
    # calclulate the Inter_quatile_range IQR
    IQR = Q3 - Q1
    # calculate the lower limit and upper  limit (LL & UL)
    LL = Q1 - 1.5 * IQR
    UL = Q3 + 1.5 * IQR
    # now filter the column to remove the outliers
    # replace all the values that are less or equal to the LL in the hours per weeek column with the LL
    df.loc[df['year'] <= LL, 'year'] = LL
    # do the same for values greater than the UL
    df.loc[df['year'] >= UL, 'year'] = UL
    return df


# removee the outliers
def remove_mileage_outliers(df):
    # calculate the Quantiles(Q1 and Q3)
    Q1 = df['Mileage'].quantile(0.25)
    Q3 = df['Mileage'].quantile(0.75)
    # calclulate the Inter_quatile_range IQR
    IQR = Q3 - Q1
    # calculate the lower limit and upper  limit (LL & UL)
    LL = Q1 - 1.5 * IQR
    UL = Q3 + 1.5 * IQR
    # now filter the column to remove the outliers
    # replace all the values that are less or equal to the LL in the hours per weeek column with the LL
    df.loc[df['Mileage'] <= LL, 'Mileage'] = LL
    # do the same for values greater than the UL
    df.loc[df['Mileage'] >= UL, 'Mileage'] = UL
    return df

# seperate the dependant and independent features
def seperate_dataset(df_copy):
    # drop the unimportant features
    df_target = pd.DataFrame()
    df_target['Price']=df_copy['Price']
    df_copy.drop(['fuel_type'], axis=1, inplace=True)
    X = df_copy.drop('Price',axis =1)
    #y = df_copy['Price']
    return X, df_target
# feature engineer the categorical features
def feature_engineering(df):
    # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
    # and place that in label encoder race as a dictionary
    #df.drop(['Unnamed: 0'], axis=1, inplace=True)
    label_encoding_name = {value: key for key, value in enumerate(df['name'].unique())}
    df['name'] = df['name'].map(label_encoding_name)

    label_encoding_company = {value: key for key, value in enumerate(df['company'].unique())}
    df['company'] = df['company'].map(label_encoding_company)    
    return df

# standardization
def feature_scaling(features):
    # get the standard scaler that was used in training
    # load the scaaler
    # generate unique id's for the prices
    features['car_ids'] = generate_uuids(len(features))
    sc = None
    scaler_path = os.path.join('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 11 - Car Price Prediction/dataset/',
    'standard_scaler.pkl')
    with open(scaler_path, 'rb') as scaler_file:
        sc = pickle.load(scaler_file)
    scaled_features = sc.transform(features)
    return scaled_features

def load_model(run_id):
    logged_model = f'http://127.0.0.1:5000/#/experiments/5/runs/{RUN_ID}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(model, features, df_target):
    # apply the model for prediction and combine the results with the old one in a single dataframe
    x = np.array(features).reshape(1, -1)
    y_pred = model.predict(x)

    df_result = pd.DataFrame()
    df_result['ride_id'] = features['ride_id']
    df_result['actual_price'] = df_target['Price']
    df_result['predicted_price'] = y_pred
    df_result['diff'] = df_result['actual_price'] - df_result['predicted_price']
    df_result['model_version'] = RUN_ID
    df_result.to_parquet(output_file, index=False)


@flow
def car_price_prediction(run_id: str, features, df_target):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    ml_model = load_model(run_id)
    scaled_features = feature_scaling(features)
    #input_file, output_file = get_paths(run_date, taxi_type, run_id)

    apply_model(model=ml_model, features=scaled_features, df_target=df_target)




@flow
def main_flow(filename = "C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv"):
    # read dataset
    df_copy = read_dataset(filename)
    print("Done reading...")
    #apply function
    df_copy = fix_price_datatype(df_copy)
    print("Done fixing price datatypes...")
    #apply function
    df_copy = fix_mileage_datatype(df_copy)
    print("Done fixing mileage datatypes...")
    # change price and mileage to float and int
    df_copy = fix_datatypes(df_copy)
    print("Done fixing datatypes to floats...")
    #apply the function
    df_copy = fixing_nans(df_copy)
    # remove outliers
    df_copy = remove_year_outliers(df_copy)
    df_copy = remove_mileage_outliers(df_copy)
    # apply function
    X, df_target = seperate_dataset(df_copy)
    # feature engineer categorical features
    prepared_features = feature_engineering(X)
    # scale the features
    scaled_features= feature_scaling(prepared_features)

    # get the model
    #ml_model = load_model(RUN_ID)

    #prediction and conceantenation of the datadframe
    car_price_prediction(run_id=RUN_ID,features=scaled_features,df_target=df_target)

    # save the correct dataframe
    #df_copy.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\preprocessed_dataset.csv')
    #send data to Feature Enginneering
    #orchest.output((df_copy, X, y),name='preprocessed-df')


if __name__ =='__main__':
    main_flow()

