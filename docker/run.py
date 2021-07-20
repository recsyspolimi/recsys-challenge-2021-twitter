#!/env/bin/python

import lightgbm as lgb
from complete_preprocess_script import do_preprocessing
from complete_feature_extraction_script import do_feature_extraction
import dask.dataframe as dd
import os
import pathlib as pl
import pandas as pd

'''
On the remote machine there will be a /test folder with the raw dataset. This will be our data_path. 
All the additional content, deriving from Preprocessing and Feature Extraction, will be placed in the /workflow folder (aka base). 
Initially, there will be only 2 subfolders:
- Dictionary: where all dicts, jsons and stuff from FE is placed
- Models: where the models will be placed
The base folder will grow while computing stuff, but during the preparation of then sub we don't care.
We just need to create a workflow folder and under it the aforementioned subfolders with correct stuff inside. 
As a peer of this of this folder, there should be the Scripts folder and the two complete-* scripts.
'''


def preprocess_dataset():

    data_path = pl.Path(__file__).parent.joinpath("test/").absolute()
    base_path = pl.Path(__file__).parent.joinpath("workflow/").absolute()
    dict_path = os.path.join(base_path, 'Dictionary')
    all_scripts = [
        "pre00_dataset_to_parquet.py",
        "pre01_map_user_id_features.py",
        "pre02_map_media_features.py",
        "pre03_map_link_id.py",
        "pre04_map_domains_id.py",
        "pre05_map_hashtags_id.py",
        "pre06_map_languages_id.py",
        "pre07_map_tweet_id.py",
        "pre08_map_tweet_type.py",
        "pre09_timestamps.py",
        "pre10_text_preprocessing.py",
        "pre20_merge_all_mapped_features.py",
        # ### "pre21_generate_subsample.py", # should not be used anymore
        # "pre22_split_train_val.py"
    ]

    config = {
        'original_dataset': os.path.join(data_path, 'part-*'),
        'base_path': os.path.join(base_path, ''),
        'temp_path': os.path.join(base_path, 'Temp'),
        'dict_path': dict_path,
        'train_val_ratio': [1, 0],
        'dask_tmp_path': os.path.join(base_path, 'Temp', 'dask_tmp'),
    }
    print(config)

    do_preprocessing(config, all_scripts, generate_dict=False, is_test=True)


def extract_features():

    base_path = pl.Path(__file__).parent.joinpath("workflow/").absolute()
    dict_path = os.path.join(base_path, 'Dictionary')
    data_path = os.path.join(base_path, 'Full_mapped_dataset')
    all_scripts = [
        'fe01_follower_features.py',
        'fe02_user_hashtags.py',
        'fe03_categorical_combo.py',
        'fe20_merge_all_features.py'
    ]

    # define all config paths needed by the subscripts
    config = {
        'data_path': data_path,
        'base_path': os.path.join(base_path, ''),
        'temp_path': os.path.join(base_path, 'Temp'),
        'preproc_dict_path': dict_path,
        'dict_path': dict_path,
        'dask_tmp_path': os.path.join(base_path, 'Temp', 'dask_tmp'),
    }
    print(config)

    do_feature_extraction(config, all_scripts, generate_dict=False, is_test=True)


def evaluate():

    base_path = pl.Path(__file__).parent.joinpath("workflow/").absolute()
    data_path = os.path.join(base_path, 'All_feature_dataset')
    model_path = os.path.join(base_path, '../Models')

    all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                    "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
                    "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                    "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
                    "engaging_user_account_creation", "engagee_follows_engager"]

    original_df = pd.concat([pd.read_csv(f, sep='\x01', names=all_features) for f in [os.path.join('./test', f)
                                                                            for f in os.listdir('./test') if 'part' in f]],
                                                                            ignore_index=True)

    column_tweet_id = original_df.tweet_id
    column_user_id = original_df.engaging_user_id

    del original_df

    df = dd.read_parquet(data_path)

    df = df.drop(columns=['tweet_timestamp',
                          'creator_creation_timestamp',
                          'engager_creation_timestamp',
                          'mapped_creator_id',
                          'mapped_engager_id',
                          'mapped_tweet_links_id',
                          'mapped_domains_id',
                          'mapped_tweet_hashtags_id',
                          'mapped_tweet_id',
                          'tweet_datetime',
                          'engager_creation_datetime',
                          'creator_creation_datetime',
                          'engager_hashtag_count_list'])

    df = df.compute()

    like_classifier = lgb.Booster(model_file=os.path.join(model_path, 'like.txt'))
    retweet_classifier = lgb.Booster(model_file=os.path.join(model_path, 'retweet.txt'))
    reply_classifier = lgb.Booster(model_file=os.path.join(model_path, 'reply.txt'))
    comment_classifier = lgb.Booster(model_file=os.path.join(model_path, 'comment.txt'))

    likes = like_classifier.predict(df)
    replies = reply_classifier.predict(df)
    comments = comment_classifier.predict(df)
    retweets = retweet_classifier.predict(df)

    predictions = pd.DataFrame({"reply_prediction": replies, "retweet_prediction": retweets,
                                "comment_prediction": comments, "like_prediction": likes})

    df_final = pd.concat([column_tweet_id, column_user_id, predictions], axis=1)

    df_final.to_csv("results.csv", header=False, index=False)


if __name__ == "__main__":
    preprocess_dataset()
    extract_features()
    evaluate()
