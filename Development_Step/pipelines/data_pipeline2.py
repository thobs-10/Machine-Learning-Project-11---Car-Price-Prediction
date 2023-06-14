from airflow import DAG
from datetime import datetime,timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.email_operator import EmailOperator


from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
from zenml.pipelines import pipeline
#from ..data_workflows_steps.data_preprocessing import read_dataset, fix_price_datatype,fix_mileage_datatype,fix_datatypes, fixing_nans
from data_workflows_steps import get_data

docker_settings = DockerSettings(required_integrations=[AIRFLOW])

default_args = {"owner":"airflow","start_date":datetime(2021,3,7)}
#with DAG(dag_id="workflow",default_args=default_args,schedule_interval='@daily') as dag:

