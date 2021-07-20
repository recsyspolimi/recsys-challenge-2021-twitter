from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import os, sys
import time
import RootPath
from Scripts.utilities import start_cluster
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers   import Dense,Activation,Dropout,Embedding,LSTM,Concatenate,Input,Flatten,BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import regularizers
from tensorflow.keras.losses import *
import numpy as np
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
import gc
import lightgbm as lgb

def buildModel(layer,inputSize,depth=3,firstHidden=256,dropout=0,reduction_factor=2,loss=BinaryCrossentropy(from_logits=False),useNormalization=True,optimizer=Adam,lr=0.0005):
    model=Sequential()
    #after first layer it gets  ignored
    shape=(inputSize,)
    
    size=firstHidden
    model.add(layer)
    for i in range(depth):
        model.add(Dense(size,input_shape=shape,activation="relu"))
        model.add(Dropout(dropout))
        if useNormalization:
            model.add(BatchNormalization())
        size=size//reduction_factor
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss=loss, metrics=[tf.keras.metrics.AUC(name="PRAUC", curve='PR'),"accuracy"],optimizer=optimizer(learning_rate=lr))
    return model

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr


def rce(y_true, y_pred):
    cross_entropy = log_loss(y_true, y_pred)
    data_ctr = calculate_ctr(y_true)
    strawman_cross_entropy = log_loss(y_true, [data_ctr for _ in range(len(y_true))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def ap(y_true, y_pred):
    return  average_precision_score(y_true, y_pred)


if __name__ == '__main__':
    print('Python %s on %s' % (sys.version, sys.platform))

    if RootPath.is_aws():
        print("Detected running on AWS!")
        c = start_cluster(n_workers=8, threads_per_worker=1, memory_limit="16GB", processes=True)

    else:
        print("Running on local")
    print(f"Dataset folder used: {RootPath.get_dataset_path()}")
    frac=0.8
    idx=3
    engCols=['engagement_reply_timestamp', 'engagement_comment_timestamp', 'engagement_retweet_timestamp','engagement_like_timestamp']
    print(engCols[idx])
    # TBD: correct the path with the real one!
    parquet_dataset_Test_path= os.path.join(RootPath.get_dataset_path(),"test")
    print('Start reading \n')
    cols=[
    'creator_follower_count',
    'creator_following_count',
    'creator_is_verified',
    'creator_creation_timestamp',
    'engager_follower_count',
    'engager_following_count',
    'engager_is_verified',
    'engager_creation_timestamp',
    'engagement_creator_follows_engager',
    'engagement_reply_timestamp',
    'engagement_retweet_timestamp',
    'engagement_comment_timestamp',
    'engagement_like_timestamp',
    'number_of_photo',
    'number_of_gif',
    'number_of_video',
    'tweet_links_count',
    'tweet_domains_count',
    'tweet_hashtags_count',
    'tweet_hashtags_unique_count',
    'mapped_language_id',
    'mapped_tweet_type',
    'tweet_timestamp_hour_sin',
    'tweet_timestamp_hour_cos',
    'tweet_timestamp_day',
    'tweet_timestamp_weekday',
    'tweet_timestamp_hour_bin',
    'tweet_timestamp_creator_account_age_bin',
    'text_is_reply',
    'text_tokens_count',
    'text_unknown_count',
    'text_special_tokens_count',
    'text_questions_count',
    'text_semantic_separation',
    'text_newline_count',
    'text_separated_count',
    'text_char_count',
    'text_asking_like',
    'text_asking_reply',
    'text_comment_related_count',
    'text_no_comment_related_count',
    'text_asking_retweet',
    'text_nsfw_count',
    'text_kpop_count',
    'text_covid_count',
    'text_sports_count',
    'text_japanesetrending_count',
    'text_anime_count',
    'text_vtuber_count',
    'text_news_count',
    'text_myanmar_count',
    'text_genshin_count',
    'text_crypto_count',
    'text_trending_count',
    'text_love_count',
    'text_slang_count',
    'text_mention_count',
    'engager_follower_quantile',
    'creator_follower_quantile',
    'creator_follower_ratio',
    'engager_follower_ratio',
    'creator_vs_engager_follower_ratio',
    'creator_vs_engager_following_ratio',
    'CE_language__timestamp_hour_bin',
    'CE_language__timestamp_hour_bin__timestamp_weekday',
    'CE_language__type',
    'CE_language__engager_follower_quantile',
    'CE_type__timestamp_weekday',
    'CE_type__timestamp_hour_bin',
    'CE_timestamp_creator_account_age_bin__engager_follower_quantile__creator_follower_quantile',
    'CE_language__presence_of_photo__presence_of_gif__presence_of_video',
    'TE_mapped_engager_id_engagement_reply',
    'TE_number_of_photo_engagement_reply',
    'TE_number_of_gif_engagement_reply',
    'TE_number_of_video_engagement_reply',
    'TE_mapped_tweet_type_engagement_reply',
    'TE_mapped_language_id_engagement_reply',
    'TE_mapped_creator_id_engagement_reply',
    'TE_mapped_tweet_links_id_1_engagement_reply',
    'TE_mapped_tweet_links_id_2_engagement_reply',
    'TE_mapped_tweet_hashtags_id_1_engagement_reply',
    'TE_mapped_tweet_hashtags_id_2_engagement_reply',
    'TE_mapped_domains_id_1_engagement_reply',
    'TE_mapped_domains_id_2_engagement_reply',
    "TE_('mapped_domains_id_1', 'mapped_language_id', 'engagement_creator_follows_engager', 'mapped_tweet_type', 'number_of_photo', 'creator_is_verified')_engagement_reply",
    'TE_tweet_links_count_engagement_reply',
    'TE_tweet_domains_count_engagement_reply',
    'TE_tweet_hashtags_count_engagement_reply',
    'TE_tweet_hashtags_unique_count_engagement_reply',
    'TE_mapped_engager_id_engagement_retweet',
    'TE_number_of_photo_engagement_retweet',
    'TE_number_of_gif_engagement_retweet',
    'TE_number_of_video_engagement_retweet',
    'TE_mapped_tweet_type_engagement_retweet',
    'TE_mapped_language_id_engagement_retweet',
    'TE_mapped_creator_id_engagement_retweet',
    'TE_mapped_tweet_links_id_1_engagement_retweet',
    'TE_mapped_tweet_links_id_2_engagement_retweet',
    'TE_mapped_tweet_hashtags_id_1_engagement_retweet',
    'TE_mapped_tweet_hashtags_id_2_engagement_retweet',
    'TE_mapped_domains_id_1_engagement_retweet',
    'TE_mapped_domains_id_2_engagement_retweet',
    "TE_('mapped_domains_id_1', 'mapped_language_id', 'engagement_creator_follows_engager', 'mapped_tweet_type', 'number_of_photo', 'creator_is_verified')_engagement_retweet",
    'TE_tweet_links_count_engagement_retweet',
    'TE_tweet_domains_count_engagement_retweet',
    'TE_tweet_hashtags_count_engagement_retweet',
    'TE_tweet_hashtags_unique_count_engagement_retweet',
    'TE_mapped_engager_id_engagement_comment',
    'TE_number_of_photo_engagement_comment',
    'TE_number_of_gif_engagement_comment',
    'TE_number_of_video_engagement_comment',
    'TE_mapped_tweet_type_engagement_comment',
    'TE_mapped_language_id_engagement_comment',
    'TE_mapped_creator_id_engagement_comment',
    'TE_mapped_tweet_links_id_1_engagement_comment',
    'TE_mapped_tweet_links_id_2_engagement_comment',
    'TE_mapped_tweet_hashtags_id_1_engagement_comment',
    'TE_mapped_tweet_hashtags_id_2_engagement_comment',
    'TE_mapped_domains_id_1_engagement_comment',
    'TE_mapped_domains_id_2_engagement_comment',
    "TE_('mapped_domains_id_1', 'mapped_language_id', 'engagement_creator_follows_engager', 'mapped_tweet_type', 'number_of_photo', 'creator_is_verified')_engagement_comment",
    'TE_tweet_links_count_engagement_comment',
    'TE_tweet_domains_count_engagement_comment',
    'TE_tweet_hashtags_count_engagement_comment',
    'TE_tweet_hashtags_unique_count_engagement_comment',
    'TE_mapped_engager_id_engagement_like',
    'TE_number_of_photo_engagement_like',
    'TE_number_of_gif_engagement_like',
    'TE_number_of_video_engagement_like',
    'TE_mapped_tweet_type_engagement_like',
    'TE_mapped_language_id_engagement_like',
    'TE_mapped_creator_id_engagement_like',
    'TE_mapped_tweet_links_id_1_engagement_like',
    'TE_mapped_tweet_links_id_2_engagement_like',
    'TE_mapped_tweet_hashtags_id_1_engagement_like',
    'TE_mapped_tweet_hashtags_id_2_engagement_like',
    'TE_mapped_domains_id_1_engagement_like',
    'TE_mapped_domains_id_2_engagement_like',
    "TE_('mapped_domains_id_1', 'mapped_language_id', 'engagement_creator_follows_engager', 'mapped_tweet_type', 'number_of_photo', 'creator_is_verified')_engagement_like",
    'TE_tweet_links_count_engagement_like',
    'TE_tweet_domains_count_engagement_like',
    'TE_tweet_hashtags_count_engagement_like',
    'TE_tweet_hashtags_unique_count_engagement_like',
    ]

    dfTest = dd.read_parquet(parquet_dataset_Test_path, engine='pyarrow', columns=cols)

    chosen=engCols[idx]
    rest=[c for c in engCols if c!=chosen]

    dfTest = dfTest.drop(columns=rest)

    # Maybe there is a smarter way? It's 01:48, I will leave these 2 lines as they are for the moment...
    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] < 0, 0)
    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] > 0, 1)
    
    print('Start y \n')
    yTest = dfTest[chosen]
    dfTest = dfTest.drop(columns=[chosen])


    dfTest = dfTest.compute()
    yTest = yTest.compute()

    colsNN=np.load("cols.npy",allow_pickle=True)[()]
    dfNN=dfTest[colsNN].copy()
    testIn=dfNN.to_numpy(copy=False)
    
    model=tf.keras.models.load_model("like.hdf5")
    pred_NN=model.predict(testIn,batch_size=4096)
    print("end_NN")

    like_classifier = lgb.Booster(model_file="lightgbm_like.txt")

    cols_when_model_builds = like_classifier.feature_name()

    
    dfTest.columns = [c.replace("'", "apice").replace(',', 'virgola').replace("(","").replace(")","").replace(' ', '_') for c in dfTest.columns]
    
    dfTest = dfTest[cols_when_model_builds]
    pred_lgb=like_classifier.predict(dfTest.to_numpy(copy=False)).reshape((1928077, 1))
    print("end_lgb")
    

    for i in range(100):
        predFinal=pred_lgb*(100-i)/100+pred_NN*i/100
        rce_score=rce( yTest,predFinal)
        ap_score=ap(yTest,predFinal)
        with open("perf_like.txt",'a+') as f:
            f.write(f"{i}\n")
            f.write(f"AP: {ap_score}  \n ")
            f.write(f"RCE {rce_score}  \n ")
