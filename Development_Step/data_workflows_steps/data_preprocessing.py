# dependancies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import orchest
import pickle

import logging

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from prefect import flow, task
from zenml.steps import Output, step

# get data from data integrity
#data = orchest.get_inputs()
#df = data['eda-df']

#global variable
filename = "C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv"
class ReadData():

    def __init__(self) -> None:
        pass
    def reading_dataset(self):
        df = pd.read_csv(filename)
        return df
    



@step
def read_dataset() -> Output(
    df_copy = pd.DataFrame
):
    try:
        readData = ReadData()
        df = readData.reading_dataset()
        #df = pd.read_csv(filename)
        # remove the unnamed column
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

        # before making any changes lets copy the  data and work with the copy
        df_copy = df.copy()
        df_copy.drop(df.index[df_copy['Price']=='Ask For Price'], inplace = True)
        # change the ask for price text value in price column and replace it with 0
        #df_copy['Price'] = np.where(df_copy['Price'] == 'Ask For Price',0,df_copy['Price']) 
        #df_copy['Price'] = df_copy['Price'].replace('Ask For Price', 0)
        return df_copy
    except Exception as e:
        logging.error(e)
        raise e


@step
def fix_price_datatype(df_copy:pd.DataFrame) -> Output(
    df_copy = pd.DataFrame
):
    for i in range(len(df_copy)):
        cur_price = df_copy.loc[i,'Price']
        if(cur_price!= 0):
            cur_price = float(cur_price.replace(',',''))
            df_copy.loc[i,'Price']=cur_price
           
    return df_copy

# fixing the mileage column
@step
def fix_mileage_datatype(df_copy:pd.DataFrame) -> Output(
    df_copy = pd.DataFrame
):
    for i in range(len(df_copy)):
        cur_price = df_copy.loc[i,'Mileage']
        if(cur_price!=0):
            cur_price = float(cur_price.replace(',',''))
            df_copy.loc[i,'Mileage']=cur_price
    return df_copy

@step
def fix_datatypes(df_copy:pd.DataFrame)-> Output(
    df_copy = pd.DataFrame
):
    # make the price column to be aa float
    df_copy['Price'] = df_copy['Price'].astype(float)
    df_copy['Mileage'] = df_copy['Mileage'].astype(float)
    return df_copy


# deal with the NAN
@step
def fixing_nans(df_copy:pd.DataFrame)-> Output(
    df_copy = pd.DataFrame
):
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    imputer.fit(df_copy.iloc[:, 3:4])
    df_copy.iloc[:, 3:4] = imputer.transform(df_copy.iloc[:, 3:4])
    return df_copy


# removee the outliers
@step
def remove_year_outliers(df:pd.DataFrame)-> Output(
    df = pd.DataFrame
):
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
@step
def remove_mileage_outliers(df:pd.DataFrame)-> Output(
    df = pd.DataFrame
):
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
@step
def seperate_dataset(df:pd.DataFrame)-> Output(
    X = pd.DataFrame,
    y = pd.Series
):
    X = df.drop('Price',axis =1)
    y = df['Price']
    return X,y

@step
def hot_encoding(X:pd.DataFrame)-> Output(
    X = np.ndarray
):
    new_X = X.iloc[:, :5].values
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    new_X = np.array(ct.fit_transform(new_X))
    return new_X

# def main_flow(filename = "C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv"):
#     # read dataset
#     df_copy = read_dataset(filename)
#     print("Done reading...")
#     #apply function
#     df_copy = fix_price_datatype(df_copy)
#     print("Done fixing price datatypes...")
#     #apply function
#     df_copy = fix_mileage_datatype(df_copy)
#     print("Done fixing mileage datatypes...")
#     # change price and mileage to float and int
#     df_copy = fix_datatypes(df_copy)
#     print("Done fixing datatypes to floats...")
#     #apply the function
#     df_copy = fixing_nans(df_copy)
#     # remove outliers
#     df_copy = remove_year_outliers(df_copy)
#     df_copy = remove_mileage_outliers(df_copy)
#     # apply function
#     X,y = seperate_dataset(df_copy)
#     # one-hot encoding
#     X = hot_encoding(X)
#     # save the correct dataframe
#     df_copy.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\preprocessed_dataset.csv')
#     #send data to Feature Enginneering
#     #orchest.output((df_copy, X, y),name='preprocessed-df')

# main_flow()
