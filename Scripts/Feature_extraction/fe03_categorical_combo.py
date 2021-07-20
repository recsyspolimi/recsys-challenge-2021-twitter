from typing import Tuple, Dict

from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from feature_extraction_utilities import preproc_dict_path, dict_path, temp_output_path, dataset_path

import numpy as np
import dask.dataframe as dd
import os
import json

N_BINS_HOURS_DAY = 8

with open(os.path.join(preproc_dict_path, 'language_id.json'), 'r') as read_file:
    language_id_info = json.load(read_file)
with open(os.path.join(preproc_dict_path, 'tweet_type.json'), 'r') as read_file:
    tweet_type_info = json.load(read_file)
n_languages = language_id_info["len"] + 1
n_tweet_types = tweet_type_info["len"] + 1


columns = [
    'tweet_timestamp_hour_bin',
    'mapped_language_id',
    'tweet_timestamp_hour',
    'tweet_timestamp_weekday',
    'mapped_tweet_type',
    'tweet_timestamp_creator_account_age_bin',
    'presence_of_photo',
    'presence_of_gif',
    'presence_of_video'
]


feature_cardinality = {
    'tweet_timestamp_hour_bin': N_BINS_HOURS_DAY,
    'mapped_language_id': n_languages,
    'tweet_timestamp_hour': 24,
    'tweet_timestamp_weekday': 7,
    'engager_follower_quantile': 5,
    'creator_follower_quantile': 5,
    'mapped_tweet_type': n_tweet_types,
    'tweet_timestamp_creator_account_age_bin': 3,
    'presence_of_photo': 2,
    'presence_of_gif': 2,
    'presence_of_video': 2,
}

to_be_combined = [
    ('mapped_language_id', 'tweet_timestamp_hour_bin'),
    ('mapped_language_id', 'tweet_timestamp_hour_bin', 'tweet_timestamp_weekday'),
    ('mapped_language_id', 'mapped_tweet_type'),
    ('mapped_language_id', 'engager_follower_quantile'),
    ('mapped_tweet_type', 'tweet_timestamp_weekday'),
    ('mapped_tweet_type', 'tweet_timestamp_hour_bin'),
    ('tweet_timestamp_creator_account_age_bin', 'engager_follower_quantile', 'creator_follower_quantile'),
    ('mapped_language_id', 'presence_of_photo', 'presence_of_gif', 'presence_of_video')
]


def result_name_given_features(features: Tuple[str, ...]) -> str:
    return "CE_" + "__".join([f.replace("tweet_", "").replace("mapped_", "").replace("_id", "") for f in features])


out_cols = [result_name_given_features(f) for f in to_be_combined]
out_frame_name = "categorical_features"

def combine_features(df: dd.DataFrame, features: Tuple[str, ...], feature_cardinality: Dict[str, int]) -> dd.Series:
    card = [feature_cardinality[f] for f in features]
    cum_card = np.cumprod([1] + card)  # to get [1, card_0, card_0 * card_1, ...]

    if cum_card[-1] < np.iinfo(np.uint8).max:
        out_type = np.uint8
    elif cum_card[-1] < np.iinfo(np.uint16).max:
        out_type = np.uint16
    elif cum_card[-1] < np.iinfo(np.uint32).max:
        out_type = np.uint32
    else:
        out_type = np.uint64
    res_name = result_name_given_features(features)
    res = sum(df[f] * c for f, c in zip(features, cum_card))
    return res.rename(res_name).astype(out_type)


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    df = read_dataset(dataset_path, columns)
    quantile_df = read_dataset(os.path.join(temp_output_path, 'user_features'), None)
    df['engager_follower_quantile'] = quantile_df['engager_follower_quantile'] # quantiles not yet added to full dataset
    df['creator_follower_quantile'] = quantile_df['creator_follower_quantile']

    # Map the columns
    out = dd.concat(
        [combine_features(df, f, feature_cardinality).to_frame() for f in to_be_combined],
        axis=1, ignore_unknown_divisions=True
    )

    # Write the output dataset
    save_dataset(temp_output_path, out, out_frame_name)
