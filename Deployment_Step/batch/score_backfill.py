from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

import score


@flow
def car_price_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2022, month=4, day=1)

    d = start_date

    while d <= end_date:
        score.car_price_prediction(
            run_id='e1efc53e9bd149078b0c12aeaa6365df',
            features="",
            df_target=""
        )

        d = d + relativedelta(months=1)


if __name__ == '__main__':
    car_price_prediction_backfill()