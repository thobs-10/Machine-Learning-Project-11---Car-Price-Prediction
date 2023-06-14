from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from score import car_price_prediction

deployment = Deployment.build_from_flow(
    flow=car_price_prediction,
    name="car_price_prediction",
    parameters={
        "run_id": "e1efc53e9bd149078b0c12aeaa6365df",
        "features":"scaled_features",
        "df_target": "df_target"
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
    work_queue_name="ml",
)

deployment.apply()
