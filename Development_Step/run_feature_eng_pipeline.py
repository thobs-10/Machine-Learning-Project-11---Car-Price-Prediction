from data_workflows_steps.feature_engineering import feature_engineering 
from data_workflows_steps.feature_engineering import seperate_dataset
from data_workflows_steps.feature_engineering import label_encode
from data_workflows_steps.feature_engineering import feature_importance
from data_workflows_steps.feature_engineering import drop_top_correlated_features
from data_workflows_steps.feature_engineering import drop_correlated
from data_workflows_steps.feature_engineering import split_for_PCA
from data_workflows_steps.feature_engineering import principal_component_analysis
from data_workflows_steps.feature_engineering import get_most_important_features
from data_workflows_steps.feature_engineering import feature_scaling
from data_workflows_steps.feature_engineering import convert_datasets


from pipelines.feature_engineering_pipeline import data_engineering_workflow

import pandas as pd

def main():
    pipeline_instance = data_engineering_workflow(
        feature_engineering(),
        seperate_dataset(),
        label_encode(),
        feature_importance(),
        drop_top_correlated_features(),
        drop_correlated(),
        split_for_PCA(),
        principal_component_analysis(),
        get_most_important_features(),
        feature_scaling(),
        convert_datasets()
    )

    pipeline_instance.run( run_name='feature_engineering_data_pipeline_dag')


if __name__ == "__main__":
    main()

