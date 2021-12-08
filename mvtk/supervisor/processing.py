import pandas as pd
import numpy as np
import public


@public.add
def replace_nulls(df, replace, column_names):
    return df.fillna({k: replace for k in column_names})


@public.add
# Normalize timestamp column values.
def normalize_ts_columns(df, column_names):
    for column_name in column_names:
        normalize_ts_column(df, column_name)
    return df


# convert timestamp in HH:mm:ss to seconds -
#   pandas timedelta takes the time format and converts them to seconds.
# divide by the result by the total number of seconds in a day.
# this normalizes the timestamp to a number between 0 and 1.
# round off the value to 5 decimal places.
@public.add
def normalize_ts_column(df, column_name):
    df[column_name] = pd.to_timedelta(
        df[column_name].dt.strftime("%H:%M:%S")
    ).dt.total_seconds()
    df[column_name] = df[column_name].replace(np.nan, -1)
    df[column_name] = df[column_name].apply(
        lambda x: round(x / 86400, 5) if x >= 0 else x
    )

    return df
