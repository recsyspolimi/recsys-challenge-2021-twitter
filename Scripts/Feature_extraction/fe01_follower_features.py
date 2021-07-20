from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from feature_extraction_utilities import preproc_dict_path, dict_path, temp_output_path, dataset_path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import time, os
import json
import gzip

dict_name = 'user_quantiles'
out_cols = ['engager_follower_quantile', 'creator_follower_quantile', 'creator_follower_ratio', 'engager_follower_ratio', 'creator_vs_engager_follower_ratio', 'creator_vs_engager_following_ratio']
out_frame_name = 'user_features'


def create_save_user_quantiles(df: dask.dataframe.DataFrame, quantiles: np.array) -> None:
    """
    Find out quantile values for followers the specified df

    Args:
        df: dataframe containing ids and follower count for engagers and creators
        quantiles: the quantiles to be found

    Returns:
        None
    """
    #Small preprocessing to guarantee structure in quantiles
    if quantiles[0] != 0:
        quantiles = np.insert(quantiles, 0, 0)
    if quantiles[-1] != 1:
        quantiles = np.append(quantiles, 1)

    start = time.time()

    print("Creating user array to count followers")
    with open(os.path.join(preproc_dict_path, 'user_id.json'), 'r') as read_file:
        user_id_info = json.load(read_file)
    n_users = user_id_info["len"] + 1  # id 0 reserved for default

    print(f"Loaded! Number of users+1[default]: {n_users}, from file: {os.path.join(dict_path, 'user_id.json')}")

    user_followers = np.zeros(n_users, dtype=np.int32)
    print(f"Allocated user_followers array! Its size: {user_followers.nbytes /10**6} MB")

    # single thread access due to numpy array not being thread-safe
    for part in df.partitions:
        temp_df = part.compute()
        user_followers[temp_df['mapped_creator_id']] = temp_df['creator_follower_count']
        user_followers[temp_df['mapped_engager_id']] = temp_df['engager_follower_count']

    print(f"Computed result! First 20 users' followers {user_followers[1:20]}")
    # remember that id=0 is to not be used
    values = np.quantile(user_followers[1:], quantiles)
    print(f"Extracted quantiles of user followers! Quantiles={quantiles}, Values={values}")

    path = os.path.join(dict_path, dict_name)
    print(f"Saving array in {path}")
    with gzip.GzipFile(path, 'wb') as file:
        np.save(file, values)  # best compression-speed tradeoff with this technique

    end = time.time()
    print(f"Done. Time elapsed: {end - start}")
    return None


def map_quantiles_over_series(series: dask.dataframe.Series, out_name: str, out_dtype=np.int8) -> dask.dataframe.Series:
    """
    Map over a follower number series a categorization depending on quantiles produced before

    Args:
        series: the original series over which map the quantiles
        out_name: the name of the output series
        out_dtype: optional type specification for the output values. By default, it is np.int8 since we have 10 classes

    Returns:
        a dask.dataframe.Series contatining the category corresponding to each value in the original series
    """
    path = os.path.join(dict_path, dict_name)
    with gzip.GzipFile(path, 'rb') as file:
        quantile_values = np.load(file, allow_pickle=True)

    # discard first and last, corresponding to min and max
    quantile_values = np.delete(quantile_values, 0)
    quantile_values = np.delete(quantile_values, -1)

    def apply_quantiles(series, q, dtype):
        cat = np.searchsorted(q, series.values)
        return pd.Series(cat, name=out_name, dtype=dtype)

    return series.map_partitions(apply_quantiles, quantile_values, out_dtype, meta=(out_name, np.int8))


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    # Load dataset
    columns = [
        'mapped_creator_id',
        'mapped_engager_id',
        'creator_follower_count',
        'creator_following_count',
        'engager_follower_count',
        'engager_following_count'
    ]
    df = read_dataset(dataset_path, columns)

    # Create Dict
    if generate_dict:
        # Custom function to create mapping
        quantiles = [0.20, 0.40, 0.60, 0.80]
        create_save_user_quantiles(df, quantiles)

    # Always generate quantile columns since used after
    # Map the columns
    out = dd.concat(
        [
            map_quantiles_over_series(df['engager_follower_count'], 'engager_follower_quantile').to_frame(),
            map_quantiles_over_series(df['creator_follower_count'], 'creator_follower_quantile').to_frame(),
            (df['creator_follower_count'] / (df['creator_following_count'] + 1)).to_frame(name='creator_follower_ratio'),
            (df['engager_follower_count'] / (df['engager_following_count'] + 1)).to_frame(name='engager_follower_ratio'),
            (df['creator_follower_count'] / (df['engager_follower_count'] + 1)).to_frame(name='creator_vs_engager_follower_ratio'),
            (df['creator_following_count'] / (df['engager_following_count'] + 1)).to_frame(name='creator_vs_engager_following_ratio'),
        ],
        axis=1, ignore_unknown_divisions=True
    )

    # Write the output dataset
    save_dataset(temp_output_path, out, out_frame_name)
