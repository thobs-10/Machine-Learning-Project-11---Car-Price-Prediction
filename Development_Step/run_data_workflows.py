
from data_workflows_steps.data_preprocessing import read_dataset 
from data_workflows_steps.data_preprocessing import fix_price_datatype
from data_workflows_steps.data_preprocessing import fix_mileage_datatype
from data_workflows_steps.data_preprocessing import fix_datatypes
from data_workflows_steps.data_preprocessing import fixing_nans
from data_workflows_steps.data_preprocessing import remove_year_outliers
from data_workflows_steps.data_preprocessing import remove_mileage_outliers
from data_workflows_steps.data_preprocessing import seperate_dataset
from data_workflows_steps.data_preprocessing import hot_encoding
from data_workflows_steps.get_data import import_data

from pipelines.data_pipeline import data_workflow

from zenml.post_execution import get_run

import pandas as pd

filename = 'C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\quikr_car.csv'
eda_filename = "C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 11 - Car Price Prediction\\dataset\\eda_dataset.csv"
# read the data from the dataset folder
df1 = pd.DataFrame()
def main():
    pipeline_instance = data_workflow(
        read_dataset(),
        fix_price_datatype(),
        fix_mileage_datatype(),
        fix_datatypes(),
        fixing_nans(),
        remove_year_outliers(),
        remove_mileage_outliers(),
        seperate_dataset(),
        hot_encoding()
    )

    pipeline_instance.run( run_name='data_pipeline_dag')


if __name__ == "__main__":
    main()
    

 
    