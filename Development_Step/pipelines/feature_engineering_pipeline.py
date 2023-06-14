from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
from zenml.pipelines import pipeline
from data_workflows_steps import get_data

docker_settings = DockerSettings(required_integrations=[AIRFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def data_engineering_workflow(
    feature_engineering,
    seperate_dataset,
    label_encode,
    feature_importance,
    drop_top_correlated_features,
    drop_correlated,
    split_for_PCA,
    principal_component_analysis,
    get_most_important_features,
    feature_scaling,
    convert_datasets
):
    df = feature_engineering()
    X,X_copy,y = seperate_dataset(df)
    X = label_encode(X)
    selection = feature_importance(X,y)
    features_to_drop, X_df = drop_top_correlated_features(X)
    X = drop_correlated(X_df, features_to_drop)
    X_train,X_test =split_for_PCA(X,y)
    df, selected_x, most_important_names = principal_component_analysis(X_train,X_test,X)
    X_train, X_test, y_train, y_test = get_most_important_features(most_important_names,X,y)
    scaled_X_train, scaled_X_test = feature_scaling(X_train, X_test)
    X_train_df,X_test_df,y_train_df,y_test_df = convert_datasets(scaled_X_train, scaled_X_test,y_train, y_test)
