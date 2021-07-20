from Scripts.Preprocessing.pre00_dataset_to_parquet import simple_features_nan
from preprocessing_utilities import output_path, temp_output_path, dataset_path
from Scripts.utilities import read_dataset, start_correct_cluster, parse_args

import dask.dataframe as dd
import os

import pre01_map_user_id_features, pre02_map_media_features, pre03_map_link_id, pre04_map_domains_id, pre05_map_hashtags_id, pre06_map_languages_id, pre07_map_tweet_id, pre08_map_tweet_type, pre09_timestamps, pre10_text_preprocessing

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False) # better False

    ### Do merging across different datasets

    if is_test:
        feats = [
            "engagement_reply_timestamp",
            "engagement_retweet_timestamp",
            "engagement_comment_timestamp",
            "engagement_like_timestamp"
        ]

        for f in feats:
            del simple_features_nan[f]

    # Read not mapped features from original dataset
    columns = list(simple_features_nan.keys())
    # columns.append("raw_feature_tweet_text_token")
    if generate_dict:
        columns.append('is_from_official_val')

    df = read_dataset(dataset_path, columns)

    # Prepare to load the datasets created previously
    df_list = []

    single_preproc_modules = [pre01_map_user_id_features, pre02_map_media_features, pre03_map_link_id,
                              pre04_map_domains_id, pre05_map_hashtags_id, pre06_map_languages_id, pre07_map_tweet_id,
                              pre08_map_tweet_type, pre09_timestamps, pre10_text_preprocessing]
    columns_dict = {}
    for module in single_preproc_modules:
        if hasattr(module, "out_frame_name"):
            columns_dict[module.out_frame_name] = module.out_cols
        else:
            columns_dict[module.out_cols[0]] = module.out_cols

    print("Columns dict:")
    print(columns_dict)

    # read all dfs created by the scripts individually and put them in a list
    print('Starting appending data with read_parquet')
    for name, cols in columns_dict.items():
        df_list.append(dd.read_parquet(os.path.join(temp_output_path, name)))

    print('Printing df_list')
    print(df_list)

    # Merge the datasets' columns -> also this might be a problem
    # print('Starting assigning columns (double for cycle)')
    # for cur_df in df_list:
    #     for col in cur_df.columns:
    #         df[col] = cur_df[col]
    out = dd.concat(
        [df] + df_list,
        axis=1, ignore_unknown_divisions=True
    )

    # Check correct dtypes
    # features = {f for dataset in columns_dict for f in columns_dict[dataset]}  # Set of features to consider
    # filtered_dtypes = {feat: dtype for (feat, dtype) in mapped_features_dtype.items() if feat in features}
    #
    # print(df.columns)
    # print(filtered_dtypes)

    # for feat, dtype in filtered_dtypes.items():
    #     df[feat] = df[feat].astype(dtype)

    out = out.repartition(partition_size="200MB")
    # Write final preprocessed dataset
    print("Writing to parquet")
    out.to_parquet(output_path, write_index=False, compression="snappy", engine="pyarrow", overwrite="True")
