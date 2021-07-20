import os
import time
import gc
import json
import pandas as pd
import dask.dataframe as dd
import dask
from Scripts.Feature_extraction.target_encoder import MTE_one_shot
from Scripts.utilities import save_dataset, read_dataset, parse_args
from Scripts.utilities import start_correct_cluster
from Scripts.Feature_extraction.feature_extraction_utilities import dataset_path, dict_path, temp_output_path, output_path

def addMeta(meta, feature, n):
    for i in range(1, n + 1):
        meta[feature + '_' + str(i)] = 'int32'
    return meta

def splitListFeature(df, columns, max):
    for col in columns:
        df[col] = df[col].apply(lambda x: [0, 0, 0, 0] if len(x) == 0 else x)
        df[col] = df[col].apply(lambda x: x[0:max] if len(x) > max else x)
        cols = []
        for i in range(1, max+1):
            cols.append(col + '_' + str(i))
        df[cols] = pd.DataFrame(
            df[col].tolist(),
            df[col].index, dtype=object
        ).fillna(0).astype('int32')
    return df

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    output_df_path = temp_output_path
    print(output_path)
    df = read_dataset(output_path, None)

    if 'raw_feature_tweet_text_token' in df.columns:
        df = df.drop('raw_feature_tweet_text_token', axis=1)
    #df = read_dataset('D:\\Giacomo\\Universita\\RecSys_Challenge_2021\\DatasetAWS\\Preprocessed\\Train', None)

    #df = df.repartition(npartitions=1000)
    meta1 = {k: df.dtypes[k] for k in df}
    meta1 = addMeta(meta1, 'mapped_tweet_links_id', 4)
    meta1 = addMeta(meta1, 'mapped_domains_id', 4)
    meta1 = addMeta(meta1, 'mapped_tweet_hashtags_id', 4)
    print(meta1)
    df = df.map_partitions(splitListFeature, ['mapped_tweet_links_id', 'mapped_domains_id', 'mapped_tweet_hashtags_id'], 4, meta=meta1)
    save_dataset(output_df_path, df, "Split_cols")

