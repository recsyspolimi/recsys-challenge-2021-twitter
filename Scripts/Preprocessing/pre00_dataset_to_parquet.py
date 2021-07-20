from preprocessing_utilities import original_dataset, dataset_path
from Scripts.utilities import start_correct_cluster, parse_args

import numpy as np
import pandas as pd
import dask.dataframe as dd
import os

all_features_dtype = {
    "raw_feature_tweet_text_token": pd.StringDtype(),
    "raw_feature_tweet_hashtags": pd.StringDtype(),
    "raw_feature_tweet_id": pd.StringDtype(),
    "raw_feature_tweet_media": pd.StringDtype(),
    "raw_feature_tweet_links": pd.StringDtype(),
    "raw_feature_tweet_domains": pd.StringDtype(),
    "raw_feature_tweet_type": pd.StringDtype(),
    "raw_feature_tweet_language": pd.StringDtype(),
    "tweet_timestamp": pd.Int64Dtype(),
    "raw_feature_creator_id": pd.StringDtype(),#pd.CategoricalDtype(),#'O',#pd.CategoricalDtype(), #pd.StringDtype(),
    "creator_follower_count": pd.Int32Dtype(),
    "creator_following_count": pd.Int32Dtype(),
    "creator_is_verified": pd.BooleanDtype(),
    "creator_creation_timestamp": pd.Int64Dtype(),
    "raw_feature_engager_id": pd.StringDtype(),#pd.CategoricalDtype(),#'O', #pd.CategoricalDtype(), #pd.StringDtype(),
    "engager_follower_count": pd.Int32Dtype(),
    "engager_following_count": pd.Int32Dtype(),
    "engager_is_verified": pd.BooleanDtype(),
    "engager_creation_timestamp": pd.Int64Dtype(),
    "engagement_creator_follows_engager": pd.BooleanDtype(),
    "engagement_reply_timestamp": pd.Int64Dtype(),
    "engagement_retweet_timestamp": pd.Int64Dtype(),
    "engagement_comment_timestamp": pd.Int64Dtype(),
    "engagement_like_timestamp": pd.Int64Dtype()
}

simple_features_nan = {
        "tweet_timestamp": -1,
        "creator_follower_count": -1,
        "creator_following_count": -1,
        "creator_is_verified": False,
        "creator_creation_timestamp": -1,
        "engager_follower_count": -1,
        "engager_following_count": -1,
        "engager_is_verified": False,
        "engager_creation_timestamp": -1,
        "engagement_creator_follows_engager": False,
        "engagement_reply_timestamp": 0,
        "engagement_retweet_timestamp": 0,
        "engagement_comment_timestamp": 0,
        "engagement_like_timestamp": 0
    }

feat_str_nan = {
    "raw_feature_tweet_hashtags": "",
    "raw_feature_tweet_id": "",
    "raw_feature_tweet_media": "",
    "raw_feature_tweet_links": "",
    "raw_feature_tweet_domains": "",
    "raw_feature_tweet_type": "",
    "raw_feature_tweet_language": "",
    "raw_feature_creator_id": "",  # pd.CategoricalDtype(),#'O',#pd.CategoricalDtype(), #pd.StringDtype(),
    "raw_feature_engager_id": "",
}

mapped_features_dtype = {
    "decoded_tweet_text_token": pd.StringDtype(),
    "mapped_tweet_hashtags": 'O',  # np.array(dtype=np.uint32)
    "mapped_tweet_id": np.uint32,
    "number_of_photo": np.uint8,
    "number_of_gif": np.uint8,
    "number_of_video": np.uint8,
    "mapped_tweet_links": 'O',  # np.array(dtype=np.uint32)
    "mapped_tweet_domains": 'O',  # np.array(dtype=np.uint32)
    "mapped_tweet_type": np.uint8,
    "mapped_tweet_language": np.uint8,
    "tweet_timestamp": np.int32,
    "mapped_creator_id": np.uint32,
    "creator_follower_count": np.int32,
    "creator_following_count": np.int32,
    "creator_is_verified": np.bool_,
    "creator_creation_timestamp": np.int32,
    "mapped_engager_id": np.uint32,
    "engager_follower_count": np.int32,
    "engager_following_count": np.int32,
    "engager_is_verified": np.bool_,
    "engager_creation_timestamp": np.int32,
    "engagement_creator_follows_engager": np.bool_,
    "engagement_reply_timestamp": np.int32,
    "engagement_retweet_timestamp": np.int32,
    "engagement_comment_timestamp": np.int32,
    "engagement_like_timestamp": np.int32
}

# def replace_negative(series: dd.Series, new_value):
#     print(f"Mapping column {series.name}, with value {new_value} of type {type(new_value)}")
#     def replace_series(s):
#         s. = new_value
#         return s
#     return series.map_partitions(replace_series, meta=("", type(new_value)))
#     # return series.map(lambda x: new_value if x < 0 else x)
#
#
# def replace_negative_target(series):
#     return replace_negative(series, np.int32(0))
#
#
# def replace_negative_creation_timestamp(series):
#     return replace_negative(series, np.int32(1.431126e+09 / 2 + 1.512318e+09 / 2))


def perform_mapping(df):
    print(f"Old types:\n {df.dtypes}")

    print("Mapping simpler features")
    for feat in simple_features_nan:
        print(feat, simple_features_nan[feat], mapped_features_dtype[feat])
        df[feat] = df[feat].fillna(simple_features_nan[feat]).astype(mapped_features_dtype[feat])

    print("Mapping strings to nan")
    for feat in feat_str_nan:
        print(feat, feat_str_nan[feat])
        df[feat] = df[feat].fillna(feat_str_nan[feat])  # astype bytes?

    print(f"New types:\n {df.dtypes}")

    return df


if __name__ == '__main__':

    print("Executing 00")

    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=True)

    ### Create intermediate parquet full dataset


    if is_test:
        feats = [
            "engagement_reply_timestamp",
            "engagement_retweet_timestamp",
            "engagement_comment_timestamp",
            "engagement_like_timestamp"
        ]

        for f in feats:
            del all_features_dtype[f]
            del mapped_features_dtype[f]
            del simple_features_nan[f]

    # Read data
    df = dd.read_csv(original_dataset,
                     sep='\x01',
                     names=list(all_features_dtype.keys()),
                     dtype=all_features_dtype,
                     blocksize="200MB"
                     )

    data_path = os.path.dirname(original_dataset)
    val_path = os.path.join(data_path, "official_valid")
    print(f"data_path={data_path}, val_path={val_path}")
    if generate_dict and os.path.exists(val_path):
        print("Appending Valid")
        df_valid = dd.read_csv(os.path.join(val_path, "part*"),
                     sep='\x01',
                     names=list(all_features_dtype.keys()),
                     dtype=all_features_dtype,
                     blocksize="200MB"
                     )
        df["is_from_official_val"] = False
        df_valid["is_from_official_val"] = True
        df = dd.concat([df, df_valid], axis=0)


    print(f"number of partitions is: {df.npartitions}")

    df = perform_mapping(df)

    median_creation_timestamp = np.int32(1.431126e+09 / 2 + 1.512318e+09 / 2)
    df["creator_creation_timestamp"] = df["creator_creation_timestamp"].mask(df["creator_creation_timestamp"] < 0,
                                                                             other=median_creation_timestamp)
    df["engager_creation_timestamp"] = df["engager_creation_timestamp"].mask(df["engager_creation_timestamp"] < 0,
                                                                             other=median_creation_timestamp)
    # df["creator_creation_timestamp"] = replace_negative_creation_timestamp(df["creator_creation_timestamp"])
    # df["engager_creation_timestamp"] = replace_negative_creation_timestamp(df["engager_creation_timestamp"])
    if not is_test:
        df["engagement_reply_timestamp"] = df["engagement_reply_timestamp"].mask(df["engagement_reply_timestamp"] < 0, other=0)
        df["engagement_retweet_timestamp"] = df["engagement_retweet_timestamp"].mask(df["engagement_retweet_timestamp"] < 0, other=0)
        df["engagement_comment_timestamp"] = df["engagement_comment_timestamp"].mask(df["engagement_comment_timestamp"] < 0, other=0)
        df["engagement_like_timestamp"] = df["engagement_like_timestamp"].mask(df["engagement_like_timestamp"] < 0, other=0)
        # df["engagement_reply_timestamp"] = replace_negative_target(df["engagement_reply_timestamp"])
        # df["engagement_retweet_timestamp"] = replace_negative_target(df["engagement_retweet_timestamp"])
        # df["engagement_comment_timestamp"] = replace_negative_target(df["engagement_comment_timestamp"])
        # df["engagement_like_timestamp"] = replace_negative_target(df["engagement_like_timestamp"])

    # Write to parquet
    df.to_parquet(dataset_path, write_index=False, compression='snappy', engine="pyarrow", overwrite="True")
