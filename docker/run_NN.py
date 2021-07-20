#!/env/bin/python

from tensorflow import keras
from complete_preprocess_script import do_preprocessing
from complete_feature_extraction_script import do_feature_extraction
from Scripts.Feature_extraction.feature_extraction_utilities import dataset_path, dict_path, temp_output_path, output_path
import dask.dataframe as dd
import os
import pathlib as pl
import pandas as pd
import numpy as np
import gc

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

    data_path = './test'
    base_path = './workflow'
    dict_path = os.path.join(base_path, 'Dictionary')
    all_scripts = [
        "pre00_dataset_to_parquet.py",
        "pre01_map_user_id_features.py",
        "pre02_map_media_features.py",
        "pre03_map_link_id.py",
        "pre04_map_domains_id.py",
        "pre05_map_hashtags_id.py",
        "pre06_map_languages_id.py",
        #"pre07_map_tweet_id.py",
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

    base_path = './workflow'
    dict_path = os.path.join(base_path, 'Dictionary')
    data_path = os.path.join(base_path, 'Full_mapped_dataset')
    all_scripts = [
        'fe01_follower_features.py',
        'fe02_user_hashtags.py',
        'fe03_categorical_combo.py',
        'fe20_merge_all_features.py',
        'fe_32a_target_encoding_split_cols.py',
        'fe_33_target_encoding_mapping.py'
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
    f = './part.0.parquet'
    print('reading parquet')
    test = pd.read_parquet(f)
    test=pd.get_dummies(test,columns=["mapped_tweet_type","mapped_language_id"])
    test = test.sample(frac=0.001)
    cols=np.load("./workflow/NN/columns.npy",allow_pickle=True)[()]
    print('loading models')
    like_classifier = keras.models.load_model("./workflow/Models/like.h5")
    retweet_classifier = keras.models.load_model("./workflow/Models/retweet.h5")
    reply_classifier = keras.models.load_model("./workflow/Models/reply.h5")
    comment_classifier = keras.models.load_model("./workflow/Models/comment.h5")

    cols_when_model_builds = [c for c in cols]
    test = test[cols_when_model_builds]
    
    summary=np.load("saved_summary/summary.npy",allow_pickle=True)[()]
    i=0
    for col in test.columns:
        test[col]=test[col].astype("float32")
        if col in summary:
            print(f"normalizing {col}")
            test[col]=(test[col]-summary[col]["mean"])/summary[col]["std"]
        if i%5==4:
            gc.collect()
        i+=1

        

    print('reading csv')
    df_tmp = dd.read_csv(BASE_DIR + '/*', sep='\x01', header=None, names=features)
    drop = [c for c in df_tmp.columns if c not in ['tweet_id', 'b_user_id']]
    df_tmp = df_tmp.drop(drop, axis=1)
    df_tmp = df_tmp.sample(frac=0.001)
    df_tmp = df_tmp.compute()
    
    likes = np.array([])
    replies = np.array([])
    comments = np.array([])
    retweets = np.array([])
    for chunk in np.array_split(test, 4):
        temp=chunk.to_numpy(copy=False)
        print('making predictions like')
        likes = np.append(likes, like_classifier.predict(temp,batch_size=2048))
        gc.collect()
        print('making prediction replies')
        replies = np.append(replies, reply_classifier.predict(temp,batch_size=2048))
        gc.collect()
        print('making prediction comments')
        comments = np.append(comments, comment_classifier.predict(temp,batch_size=2048))
        gc.collect()
        print('making prediction retweets')
        retweets = np.append(retweets, retweet_classifier.predict(temp,batch_size=2048))
        gc.collect()
    
    print('Predicting...')
    # print(likes, likes[0], {type(x) for x in likes})
    i = 0
    with open('results.csv', 'w') as output:
        for index, row in df_tmp.iterrows():
            reply_pred_i = replies[i]
            retweet_pred_i = retweets[i]
            retweet_comment_pred_i = comments[i]
            like_pred_i = likes[i]
            tw_id = row['tweet_id']
            u_id = row['b_user_id']
            output.write(
                f'{tw_id},{u_id},{reply_pred_i},{retweet_pred_i},{retweet_comment_pred_i},{like_pred_i}\n')
            i = i + 1


if __name__ == "__main__":
    features = [
        'text_tokens',  ###############
        'hashtags',  # Tweet Features
        'tweet_id',  #
        'media',  #
        'links',  #
        'domains',  #
        'tweet_type',  #
        'language',  #
        'timestamp',  ###############
        'a_user_id',  ###########################
        'a_follower_count',  # Engaged With User Features
        'a_following_count',  #
        'a_is_verified',  #
        'a_account_creation',  ###########################
        'b_user_id',  #######################
        'b_follower_count',  # Engaging User Features
        'b_following_count',  #
        'b_is_verified',  #
        'b_account_creation',  #######################
        'b_follows_a',  ####################
    ]




    BASE_DIR = './test'
    #preprocess_dataset()
    #extract_features()
    evaluate()
