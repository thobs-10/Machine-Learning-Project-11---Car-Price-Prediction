from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
from zenml.pipelines import pipeline
#from ..data_workflows_steps.data_preprocessing import read_dataset, fix_price_datatype,fix_mileage_datatype,fix_datatypes, fixing_nans
from data_workflows_steps import get_data

#docker_settings = DockerSettings(required_integrations=[AIRFLOW])
# docker_settings = DockerSettings(
#     required_integrations=[AIRFLOW], requirements=["torchvision"]
# )
# global variables
filename = 'C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\quikr_car.csv'
eda_filename = "C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv"

#read_data = read_dataset()
#fix_price_datatype = fix_price_datatype()
@pipeline(enable_cache=False)
def data_workflow(
    read_dataset,
    fix_price_datatype,
    fix_mileage_datatype,
    fix_datatypes,
    fixing_nans,
    remove_year_outliers,
    remove_mileage_outliers,
    seperate_dataset,
    hot_encoding
):
     
    # the steps for the DAG 
    # get data from get_data script
    #get_data()
    df = read_dataset()
    df = fix_price_datatype(df)
    df = fix_mileage_datatype(df)
    df = fix_datatypes(df)
    df = fixing_nans(df)
    df = remove_year_outliers(df)
    df = remove_mileage_outliers(df)
    X, y = seperate_dataset(df)
    X = hot_encoding(X)

   

