from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import temp_output_path, dataset_path

import numpy as np
from datetime import datetime
import pandas as pd
import dask.dataframe as dd

out_cols = ['tweet_timestamp_hour_sin', 'tweet_timestamp_hour_cos', 'tweet_timestamp_day', 'tweet_timestamp_month',
            'tweet_timestamp_weekday', 'tweet_timestamp_hour_bin', 'tweet_timestamp_creator_account_age',
            'creator_creation_datetime', 'engager_creation_datetime', 'tweet_datetime', 'tweet_timestamp_creator_account_age_bin']
out_frame_name = 'tweet_timestamp_features'

N_BINS_HOURS_DAY = 8 # number of bins for the hours in timestamp


def apply_quantiles(series, q, dtype):
    cat = np.searchsorted(q, series.values)
    return pd.Series(cat, dtype=dtype)


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=True)

    # Load dataset
    columns = [
        "tweet_timestamp",
        "creator_creation_timestamp",
        "engager_creation_timestamp"
    ]
    print(dataset_path)
    df = read_dataset(dataset_path, columns)

    datetime_tweet = df["tweet_timestamp"].map(lambda x: datetime.fromtimestamp(x))
    datetime_creator = df["creator_creation_timestamp"].map(lambda x: datetime.fromtimestamp(x))
    datetime_engager = df["engager_creation_timestamp"].map(lambda x: datetime.fromtimestamp(x))

    hours = datetime_tweet.map(lambda x: x.hour)
    constant = np.float32(np.pi / 24)
    hours_sin = hours.map_partitions(lambda x: np.sin(x * constant))
    hours_cos = hours.map_partitions(lambda x: np.cos(x * constant))

    # IMPORTANT: IF YOU EVER CHANGE THE QUANTILES, CHECK THAT EVERYTHING IS OK IN FE_03_CATEGORICAL_COMBO'S FEATURE_CARDINALITY
    hours_per_bin = 24 // N_BINS_HOURS_DAY
    day_quantiles = np.arange(hours_per_bin, 24,
                              hours_per_bin)  # for N_BINS_HOURS_DAY=3  0-6:night -> 0, 6-12: morning -> 1, 12-18: afternoon -> 2, 18-24: night -> 3
    age_quantiles = np.array([12, 40])  # 0-12:young -> 0, 12-40:middle ->1, 40+: veteran -> 3

    creator_account_age = (datetime_tweet - datetime_creator) // np.timedelta64(1, 'M')

    out = dd.concat(
        [
            datetime_tweet.to_frame(name='tweet_datetime'),
            datetime_engager.to_frame(name='engager_creation_datetime'),
            datetime_creator.to_frame(name='creator_creation_datetime'),
            hours.to_frame(name='tweet_timestamp_hour'),
            hours_sin.to_frame(name='tweet_timestamp_hour_sin'),
            hours_cos.to_frame(name='tweet_timestamp_hour_cos'),
            datetime_tweet.map(lambda x: x.day).astype(np.int8).to_frame(name='tweet_timestamp_day'),
            datetime_tweet.map(lambda x: x.month).astype(np.int8).to_frame(name='tweet_timestamp_month'),
            datetime_tweet.map(lambda x: x.weekday()).astype(np.int8).to_frame(name='tweet_timestamp_weekday'),

            hours.map_partitions(apply_quantiles, q=day_quantiles, dtype=np.int8).to_frame(name='tweet_timestamp_hour_bin'),

            creator_account_age.to_frame(name='tweet_timestamp_creator_account_age'),
            creator_account_age.map_partitions(apply_quantiles, q=age_quantiles, dtype=np.int8).to_frame(name='tweet_timestamp_creator_account_age_bin'),
        ],
        axis=1, ignore_unknown_divisions=True
    )

    save_dataset(temp_output_path, out, out_frame_name)
