from feature_extraction_utilities import output_path, temp_output_path, dataset_path
from Scripts.utilities import read_dataset, start_correct_cluster, parse_args

import dask.dataframe as dd
import os

from Scripts.Feature_extraction import fe01_follower_features, fe02_user_hashtags, fe03_categorical_combo
# from Scripts.Feature_extraction import fe04_1_engager_hashtags_count_LIKE, fe04_2_engager_hashtags_count_REPLY
# from Scripts.Feature_extraction import fe04_3_engager_hashtags_count_RETWEET, fe04_4_engager_hashtags_count_COMMENT
# from Scripts.Feature_extraction import fe04_5_engager_hashtags_count_POSITIVE, fe04_6_engager_hashtags_count_NEGATIVE

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Do merging across different datasets

    df = read_dataset(dataset_path, columns=None)

    # Prepare to load the datasets created previously
    df_list = []

    single_preproc_modules = [fe01_follower_features, fe02_user_hashtags, fe03_categorical_combo,
                              # fe04_1_engager_hashtags_count_LIKE, fe04_2_engager_hashtags_count_REPLY,fe04_3_engager_hashtags_count_RETWEET, fe04_4_engager_hashtags_count_COMMENT, fe04_5_engager_hashtags_count_POSITIVE, fe04_6_engager_hashtags_count_NEGATIVE
                              ]
    # columns_dict = {}


    # print("Columns dict:")
    # print(columns_dict)

    # read all dfs created by the scripts individually and put them in a list
    print('Starting appending data with read_parquet')
    for module in single_preproc_modules:
        if hasattr(module, "out_frame_name"):
            name = module.out_frame_name
        else:
            name = module.out_cols[0]
        df_list.append(dd.read_parquet(os.path.join(temp_output_path, name)))

    print('Printing df_list')
    print(df_list)

    # Merge the datasets' columns -> also this might be a problem
    print('Starting assigning columns (double for cycle)')
    # for cur_df in df_list:
    #     for col in cur_df.columns:
    #         df[col] = cur_df[col]
    out = dd.concat(
        [df] + df_list,
        axis=1, ignore_unknown_divisions=True
    )

    # Add TE
    # df_TE = dd.read_parquet(os.path.join(temp_output_path, 'TE_cols'))
    # df = df.merge(df_TE, on=['tweet_timestamp', 'mapped_creator_id', 'mapped_engager_id', 'mapped_tweet_id'],
    #               how='left')

    # Check correct dtypes
    # features = {f for dataset in columns_dict for f in columns_dict[dataset]}  # Set of features to consider
    # filtered_dtypes = {feat: dtype for (feat, dtype) in mapped_features_dtype.items() if feat in features}
    #
    # print(df.columns)
    # print(filtered_dtypes)

    # for feat, dtype in filtered_dtypes.items():
    #     df[feat] = df[feat].astype(dtype)

    #df = df.repartition(partition_size="200MB")
    # Write final preprocessed dataset
    print("Writing to parquet")
    out.to_parquet(output_path, write_index=False, compression="snappy", engine="pyarrow", overwrite="True")