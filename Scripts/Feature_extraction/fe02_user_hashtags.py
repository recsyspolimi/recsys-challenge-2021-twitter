from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from feature_extraction_utilities import dict_path, temp_output_path, dataset_path, preproc_dict_path

import dask
import dask.dataframe as dd
from dask.distributed import get_worker
import numpy as np
import pandas as pd
import time, os
import json
import gzip
import scipy.sparse as sps
import pyarrow as pa

#out_cols = ['engager_hashtag_count_list', 'engager_hashtag_count_max', 'engager_hashtag_count_avg', 'engager_language_count', 'engager_photo_count'
#            'engager_gif_count', 'engager_video_count', 'engager_type_count', 'language_hourbin_count', 'engager_weekday_count']
out_frame_name = 'counts'
N_BINS_HOURS_DAY = 8

len_vec = np.vectorize(len)


def custom_explode(row_values, col_values):
    lens = len_vec(col_values)
    users = np.repeat(row_values, lens)
    vals = np.concatenate(col_values)
    return users, vals


def data_to_csr(row_values, col_values, n_rows, n_cols, col_to_explode):
    if col_to_explode:
        row_idx, col_idx = custom_explode(row_values, col_values)
        res = sps.csr_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(n_rows, n_cols))
    else:
        row_idx = row_values
        col_idx = col_values
        res = sps.csr_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(n_rows, n_cols))
    return res


def fetch_coord_csr(m: sps.csr_matrix, row: int, col: int):
    start = m.indptr[row]
    end = m.indptr[row+1]
    row_indices = m.indices[start:end]
    if start != end:
        idx = np.searchsorted(row_indices,col)
        if idx < row_indices.size and row_indices[idx] == col:
            # element is present
            return m.data[start+idx]
    #element not present
    return 0


def compute_and_save_counts(df, row_feature_name, col_feature_name, out_name, n_rows, n_cols, col_to_explode):
    # perform summation
    # BASIC SOLUTION, FASTER ONE AFTER
    # res = sps.csr_matrix((n_rows, n_cols), dtype=np.int64)
    # for d in df.partitions:
    #     row_values, col_values = dask.compute(d[row_feature_name].values, d[col_feature_name].values)
    #     res += data_to_csr(row_values, col_values, n_rows, n_cols, col_to_explode=col_to_explode)
    print(f"Computing counts for {row_name}, {col_name}")

    def f_reduction_chunk(df, row_name, col_name, n_rows, n_cols, filter_col):
        if filter_col:
            cond = df[filter_col] > 0
            row_series = df[row_name][cond]
            col_series = df[col_name][cond]
            row_values, col_values = row_series.values, col_series.values
        else:
            row_values, col_values = df[row_name].values, df[col_name].values

        return data_to_csr(row_values, col_values, n_rows, n_cols, col_to_explode=col_to_explode)

    path = os.path.join(dict_path, out_name)
    def do_one_count(fc, appendix):
        res = df.reduction(chunk=f_reduction_chunk,
                     chunk_kwargs={'row_name': row_feature_name, 'col_name': col_feature_name,
                                   'n_rows': n_rows, "n_cols": n_cols, "filter_col": fc},
                     aggregate=lambda s: s.sum(), meta='O').compute()
        with open(path + appendix, 'wb') as file:
            sps.save_npz(file, res, compressed=True)

    for fc, appendix in zip([None, "engagement_reply_timestamp", "engagement_retweet_timestamp", "engagement_comment_timestamp",
                             "engagement_like_timestamp"], ['_all', '_reply','_retweet', '_comment', '_like']):
        do_one_count(fc, appendix)


    # results = [
    #     df.reduction(chunk=f_reduction_chunk,
    #                  chunk_kwargs={'row_name': row_feature_name, 'col_name': col_feature_name,
    #                                'n_rows': n_rows, "n_cols": n_cols, "filter_col": fc},
    #                  aggregate=lambda s: s.sum(), meta='O').compute()
    #     for fc in [None, "engagement_reply_timestamp", "engagement_retweet_timestamp", "engagement_comment_timestamp", "engagement_like_timestamp"]
    # ]
    #
    # #res = dask.compute(*results)
    # # do some prints
    # print(f"Counting finished! Final CSR matrix has shape {res[0].shape} and {res[0].nnz} elements")
    # # save
    # path = os.path.join(dict_path, out_name)
    # print(f"Saving resulting count matrix in {path}")
    # for res, appendix in zip(res, ['_all', '_reply','_retweet', '_comment', '_like']):
    #     with open(path + appendix, 'wb') as file:
    #         sps.save_npz(file, res, compressed=True)


def load_counts_and_map(c: dask.distributed.Client, df: dd.DataFrame, row_feature_name: str, col_feature_name: str, out_name: str, col_to_explode: bool, appendix: str):
    out_name = out_name + appendix
    path = os.path.join(dict_path, out_name)  # dict has same name as feature

    def load_dict(dask_worker):
        with open(path, 'rb') as file:
            counts = sps.load_npz(file)
        dask_worker.counts = counts
        return counts.shape

    r2 = c.run(load_dict, wait=True)
    print(f"Dict with counts loaded: {r2}")

    if not col_to_explode:
        print(f"Mapping features {row_feature_name}, {col_feature_name} using dict")

        def do_mapping(df, row_name, col_name, out_name):
            row_values, col_values = df[row_name].values, df[col_name].values
            w = get_worker()
            res = w.counts[row_values, col_values]  # returns a np.matrix
            res = np.asarray(res).flatten()
            return pd.DataFrame({out_name: res})

        return df.map_partitions(do_mapping, row_feature_name, col_feature_name, out_name, meta={out_name: np.int64})

    else:
        print(f"Mapping features {row_feature_name}, {col_feature_name} (to be splitted) using dict")

        def map_over_list(row, list_column):
            if list_column:
                w = get_worker()
                res = [fetch_coord_csr(w.counts, row, col) for col in list_column]
            else:
                res = []
            return res
        # no speedup possible in this case due to the variability of a column list
        map_over_list_vec = np.vectorize(map_over_list, otypes=[object])

        def my_max(x):
            return max(x) if x != [] else 0
        my_max_vec = np.vectorize(my_max)

        def my_avg(x):
            return sum(x) / len(x) if x != [] else 0
        my_avg_vec = np.vectorize(my_avg)

        def do_mapping_to_split(df, row_name, col_name, out_name):
            row_values, col_values = df[row_name].values, df[col_name].values
            res = map_over_list_vec(row_values, col_values)
            res_max = my_max_vec(res)
            res_avg = my_avg_vec(res)
            return pd.DataFrame({out_name+"_list": res, out_name+"_avg":res_avg, out_name+"_max":res_max})

        return df.map_partitions(do_mapping_to_split, row_feature_name, col_feature_name, out_name, meta={out_name+"_list": 'O', out_name+"_avg": np.float_, out_name+"_max":np.int64})


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=True)

    with open(os.path.join(preproc_dict_path, "user_id.json"), "r") as read_file:
        user_id_info = json.load(read_file)
    n_users = user_id_info["len"] + 1

    with open(os.path.join(preproc_dict_path, "hashtags_id.json"), "r") as read_file:
        hashtags_id_info = json.load(read_file)
    n_hashtags = hashtags_id_info["len"] + 1

    with open(os.path.join(preproc_dict_path, "language_id.json"), "r") as read_file:
        language_id_info = json.load(read_file)
    n_languages = language_id_info["len"] + 1

    with open(os.path.join(preproc_dict_path, 'tweet_type.json'), 'r') as read_file:
        tweet_type_info = json.load(read_file)
    n_tweet_types = tweet_type_info["len"] + 1

    # Load dataset
    columns = [
        "mapped_engager_id",
        "mapped_tweet_hashtags_id",
        "tweet_timestamp_hour_bin",
        "mapped_language_id",
        "tweet_timestamp_hour",
        "tweet_timestamp_weekday",
        "mapped_tweet_type",
        "tweet_timestamp_creator_account_age_bin",
        "presence_of_photo",
        "presence_of_gif",
        "presence_of_video",
        "engagement_reply_timestamp",
        "engagement_retweet_timestamp",
        "engagement_comment_timestamp",
        "engagement_like_timestamp"
    ]
    df = read_dataset(dataset_path, columns)

    df["presence_of_photo"] = df["presence_of_photo"].astype(np.int8)
    df["presence_of_gif"] = df["presence_of_gif"].astype(np.int8)
    df["presence_of_video"] = df["presence_of_video"].astype(np.int8)

    feature_cardinality = {
        "mapped_engager_id": n_users,
        "mapped_tweet_hashtags_id": n_hashtags,
        'tweet_timestamp_hour_bin': N_BINS_HOURS_DAY,
        'mapped_language_id': n_languages,
        'tweet_timestamp_hour': 24,
        'tweet_timestamp_weekday': 7,
        'mapped_tweet_type': n_tweet_types,
        'tweet_timestamp_creator_account_age_bin': 3,
        'presence_of_photo': 2,
        'presence_of_gif': 2,
        'presence_of_video': 2,
    }

    to_be_counted = [
        ("mapped_engager_id", "mapped_tweet_hashtags_id", True, "engager_hashtag_count"),
        ("mapped_engager_id", "mapped_language_id", False, "engager_language_count"),
        ("mapped_engager_id", "presence_of_photo", False, "engager_photo_count"),
        ("mapped_engager_id", "presence_of_gif", False, "engager_gif_count"),
        ("mapped_engager_id", "presence_of_video", False, "engager_video_count"),
        ("mapped_engager_id", "mapped_tweet_type", False, "engager_type_count"),
        ("mapped_language_id", "tweet_timestamp_hour_bin", False, "language_hourbin_count"),
        ("mapped_engager_id", "tweet_timestamp_weekday", False, "engager_weekday_count"),
    ]

    # Create Dict
    if generate_dict:
        # Custom function to create mapping
        for row_name, col_name, to_explode, out_name in to_be_counted:
            n_rows = feature_cardinality[row_name]
            n_cols = feature_cardinality[col_name]
            compute_and_save_counts(df, row_feature_name=row_name, col_feature_name=col_name, out_name=out_name, n_rows=n_rows, n_cols=n_cols, col_to_explode=to_explode)

    # avoid mapping if not on valid
    # Map the columns
    # out = dd.concat(
    #     [
    #         load_counts_and_map(c, df, row_feature_name=row_name,
    #                             col_feature_name=col_name, out_name=out_name,
    #                             col_to_explode=to_explode)
    #         for row_name, col_name, to_explode, out_name in to_be_counted
    #     ],
    #     axis=1, ignore_unknown_divisions=True
    # )

    #print(out.head(10))
    # Write each output dataset separatedly
    # in this way we get a sequential loading of one dict at a time in the worker
    for appendix in ['_all', '_reply','_retweet', '_comment', '_like']:
        for row_name, col_name, to_explode, out_name in to_be_counted:
            partial_out = load_counts_and_map(c, df, row_feature_name=row_name,
                                col_feature_name=col_name, out_name=out_name,
                                col_to_explode=to_explode, appendix=appendix)
            if "engager_hashtag_count" +appendix + "_list" in partial_out.columns:
                schema = {"engager_hashtag_count" +appendix + "_list": pa.list_(pa.uint64())}
            else:
                schema=None
            save_dataset(temp_output_path, partial_out, out_name+appendix, schema=schema)
    # do a mini merge of all generated features
    out = dd.concat(
        [
            read_dataset(os.path.join(temp_output_path, out_name+appendix), columns=None)
            for _, _, _, out_name in to_be_counted for appendix in ['_all', '_reply', '_retweet', '_comment', '_like']
        ],
        axis=1, ignore_unknown_divisions=True
    )
    save_dataset(temp_output_path, out, out_frame_name, schema={"engager_hashtag_count" +appendix + "_list": pa.list_(pa.uint64()) for appendix in ['_all', '_reply','_retweet', '_comment', '_like']})
