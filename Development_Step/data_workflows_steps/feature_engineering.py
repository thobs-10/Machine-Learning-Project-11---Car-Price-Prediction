# dependancies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import orchest
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from feature_engine.selection import DropCorrelatedFeatures
from sklearn.decomposition import PCA


from prefect import flow, task
from zenml.steps import Output, step

# get data from data preprocessing
#data = orchest.get_inputs()
#df, X, y = data['preprocessed-df']

# preprocessed dataset without one-hot encoding
filename = pd.read_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\preprocessed_dataset.csv')
class Read_Processed_Dataset():

    def __init__(self) -> None:
        pass
    def reading_dataframe(self):
        df = pd.read_csv(filename)
        return df

@step
def feature_engineering() -> Output(
    df = pd.DataFrame
):
    # get the dataframe
    try:
        processed_dataset = Read_Processed_Dataset()
        df = processed_dataset.reading_dataframe()
        # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
        # and place that in label encoder race as a dictionary
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        label_encoding_name = {value: key for key, value in enumerate(df['name'].unique())}
        df['name'] = df['name'].map(label_encoding_name)

        label_encoding_company = {value: key for key, value in enumerate(df['company'].unique())}
        df['company'] = df['company'].map(label_encoding_company)
        return df
    except Exception as e:
        logging.error(e)
        raise e    

#df = feature_engineering(df)

@step
def seperate_dataset(df_copy:pd.DataFrame) -> Output(
        X = pd.DataFrame,
        copy_X = pd.DataFrame,
        y = pd.Series
):
    X = df_copy.drop('Price',axis =1)
    y = df_copy['Price']
    # make an original copy of X features( the ones without feature engineering)
    copy_X = X.copy()
    return X,copy_X,y

#original_X, y = seperate_dataset(df)

#copy_X = original_X.copy()
@step
def label_encode(original_X:pd.DataFrame) -> Output(
    original_X = pd.DataFrame
):
    le = LabelEncoder()
    original_X['fuel_type'] = le.fit_transform(original_X['fuel_type'])
    return original_X

#original_X = label_encode(original_X)

@step
def feature_importance(X:pd.DataFrame,y:pd.Series)->Output(
    selection=ExtraTreesRegressor
):
    # Important feature using ExtraTreesRegressor
    selection = ExtraTreesRegressor()
    selection.fit(X, y)
    return selection

#selection = feature_importance(original_X, y)
@step
def drop_top_correlated_features(original_X:pd.DataFrame)->Output(
    to_drop = list,
    df_X = pd.DataFrame
):
    # removing correlated variables from dataframe using DropCorrelatedFeatures
    original_X[['name','company','year','fuel_type','Mileage']].corr()

    # removing correlated features
    df_X = pd.DataFrame(original_X)

    cor_matrix = df_X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
    return to_drop, df_X

#to_drop, df_X = drop_top_correlated_features(original_X)

# drop the two columns
@step
def drop_correlated(df:pd.DataFrame, to_drop:list)-> Output(
    X = pd.DataFrame
):
    for col in df.columns:
        for i in to_drop:
            if col == i:
                df.drop(col,axis=1,inplace = True)
    df_final = df.copy()
    X = pd.DataFrame(df_final)
    return X

#df_X_final = drop_correlated(df_X, to_drop)

# make the final dataset of X be a dataframe 
#X = pd.DataFrame(df_X_final)

# here we split the data into train and test so that we can have the required inputs to do PCA and get the most 
# important features
@step
def split_for_PCA(X:pd.DataFrame, y:pd.Series)->Output(
    X_train = np.ndarray,
    X_test = np.ndarray
):
    # original y, Latest X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train,X_test

@step
def principal_component_analysis(X_train:np.ndarray, X_test:np.ndarray, X:pd.DataFrame)->Output(
    df = pd.DataFrame,
    selected_x = pd.DataFrame,
    most_important_names = list
):

    pca = PCA(n_components = 4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # number of components
    n_pcs= pca.components_.shape[0]
    #n_pcs

    # get the index of the most important feature on EACH component i.e. largest absolute value
    # using LIST COMPREHENSION HERE
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    #most_important

    initial_feature_names = ['name','company','year','fuel_type','Mileage']

    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # using LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(sorted(dic.items()))
    #df

    # get all the selected features from the dataframe
    X_new = df.iloc[:, -1].values

    list(X_new)

    # get selected features of the PCA
    selected_x = X.loc[:, list(X_new)]
    return df, selected_x, most_important_names

#df, selected_x, most_important_names = principal_component_analysis(X_train, X_test)
@step
def get_most_important_features(most_important_names:list, X:pd.DataFrame, y:pd.Series)->Output(
    X_train = np.ndarray,
    X_test = np.ndarray,
    y_train =np.ndarray,
    y_test =np.ndarray
):
    # the names of chosen X are based on the selected features of x that have the most impact
    chosen_X = X[most_important_names]
    #chosen_X = X[['Mileage', 'name', 'company', 'year']]
    # the final spliting of data to train and test based on the most important features of X
    X_train, X_test, y_train, y_test = train_test_split(chosen_X, y, test_size = 0.2, random_state = 1)
    return X_train, X_test, y_train, y_test


# standardization
@step
def feature_scaling(X_train:np.ndarray, X_test:np.ndarray)->Output(
    X_train = np.ndarray,
    X_test = np.ndarray
):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

#X_train, X_test = feature_scaling(X_train, X_test)
@step
def convert_datasets(X_train:np.ndarray, X_test:np.ndarray,y_train:np.ndarray,y_test:np.ndarray)->Output(
    X_train_df = pd.DataFrame(),
    X_test_df = pd.DataFrame(),
    y_train_df = pd.DataFrame(),
    y_test_df = pd.DataFrame()
):
    #convert  the splitted ddataset into dataframes
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    return X_train_df,X_test_df,y_train_df,y_test_df




# save the dATA
# X_train_df.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\X_train.csv')
# X_test_df.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\X_test_df.csv')
# y_train_df.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\y_train_df.csv')
# y_test_df.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\y_test_df.csv')

#X_train.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\X_train.csv')
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
