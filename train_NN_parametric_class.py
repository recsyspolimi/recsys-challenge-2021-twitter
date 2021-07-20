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

def buildModel(layer,inputSize,depth=3,firstHidden=256,dropout=0,reduction_factor=2,loss=BinaryCrossentropy(from_logits=False),useNormalization=True,optimizer=Adam,lr=0.0003):
    model=Sequential()
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

    #code to automatically choose aws or local runtime

    if RootPath.is_aws():
        print("Detected running on AWS!")
        #set in a way that memory limit * n_workers <= max available ram and avoid memory_limit<16gb
        c = start_cluster(n_workers=16, threads_per_worker=1, memory_limit="48GB", processes=True)

    else:
        print("Running on local")

    dataset_volume_path = '/home/ubuntu/new'

    print(f"Dataset folder used: {RootPath.get_dataset_path()}")
    #change to modify percentage of data used for train-validation-test (1=100%)
    frac=1

    #choose interaction(index in the array engCols (engagement Columns))
    idx=3
    engCols=['engagement_reply_timestamp', 'engagement_comment_timestamp', 'engagement_retweet_timestamp','engagement_like_timestamp']
    print(engCols[idx])

    
    parquet_dataset_path = os.path.join(dataset_volume_path,"train")
    parquet_dataset_Test_path= os.path.join(dataset_volume_path,"test")
    

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
    'is_from_official_val',
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
    
    #load datasets
    print('Start reading \n')
    df = dd.read_parquet(parquet_dataset_path, engine='pyarrow', columns=cols)
    dfTest = dd.read_parquet(parquet_dataset_Test_path, engine='pyarrow', columns=cols)

    #choose fraction of dataset to use
    df = df.sample(frac = frac)

    chosen=engCols[idx]
    rest=[c for c in engCols if c!=chosen]

    # Drop other engagements
    df = df.drop(columns=rest)
    dfTest = dfTest.drop(columns=rest)

    #prepare output
    df[chosen] = df[chosen].mask(df[chosen] < 0, 0)
    df[chosen] = df[chosen].mask(df[chosen] > 0, 1)
    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] < 0, 0)
    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] > 0, 1)
    
    #prepare output and drop from dataset
    yTest = dfTest[chosen]
    dfTest = dfTest.drop(columns=[chosen])

    y = df[chosen]
    df = df.drop(columns=[chosen])

    print('Start compute \n')
    # From Dask to Pandas train
    df=df.astype(np.float32)
    df = df.compute()
    y = y.compute()

    print('Start compute \n')
    # From Dask to Pandas validation
    dfTest=dfTest.astype(np.float32)    
    dfTest = dfTest.compute()
    yTest = yTest.compute()
    #save list of columns and their order for inference time
    np.save("cols.npy",df.columns)
    
    yTest=yTest.to_numpy(copy=False)
    gc.collect()

    #Prepare Normalization layer to normalize NN inputs
    layer = preprocessing.Normalization()
    layer.adapt(df)
    
    print('Columns name:', df.columns)

    #rename to easier names
    X_train=df
    y_train=y
    
    #build model using normalization layer
    model = buildModel(layer,len(df.columns))
    del df, y

    BS=4096
    #prepare input and output as numpy arrays
    trainIn=X_train.to_numpy(copy=False)
    trainOut=y_train.to_numpy(copy=False)
    best=0
    #iteratively train one epoch at the time and evaluation of metrics on validation set at each step
    #model saved only on rce score improvements
    for i in range(30):

        model.fit(trainIn,trainOut,epochs=i+1,initial_epoch=i,batch_size=BS)

        preds=model.predict(dfTest.to_numpy(copy=False),batch_size=4096)
        #this line avoids exact 0 or 1 predictions which in case of mistake can lead to -infinite rce
        preds=np.clip(preds,np.finfo(float).eps,0.9999999)

        rce_score=rce( yTest,preds)
        ap_score=ap(yTest,preds)
        with open(f"perf_{chosen.replace('engagement_','').replace('_timestamp','')}.txt","a+") as f:
            f.write(f'The model scored a TEST RCE of: {rce_score}\n')
            f.write(f'The model scored an TEST AP of: {ap_score}\n')  
        if rce_score>best:
            model.save(f"{chosen.replace('engagement_','').replace('_timestamp','')}_epoch_{i}")
            best=rce_score 
