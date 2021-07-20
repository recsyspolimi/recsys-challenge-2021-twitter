from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import os, sys
import time
import lightgbm as lgb
import RootPath
from Scripts.utilities import start_cluster
import gc
from Scripts.optimizer import tune_lightGBM_sequential

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr


def rce(y_true, y_pred):
    cross_entropy = log_loss(y_true, y_pred)
    data_ctr = calculate_ctr(y_true)
    strawman_cross_entropy = log_loss(y_true, [data_ctr for _ in range(len(y_true))])
    return 'RCE', (1.0 - cross_entropy/strawman_cross_entropy)*100.0, False


def ap(y_true, y_pred):
    return 'AP', average_precision_score(y_true, y_pred), False


if __name__ == '__main__':
    print('Python %s on %s' % (sys.version, sys.platform))

    #choose interaction(index in the array engCols (engagement Columns))
    idx=0
    engCols=['engagement_like_timestamp', 'engagement_retweet_timestamp','engagement_reply_timestamp', 'engagement_comment_timestamp']

    if RootPath.is_aws():
        print("Detected running on AWS!")
        c = start_cluster(n_workers=16, threads_per_worker=1, memory_limit="24GB", processes=True)

    else:
        print("Running on local")
    print(f"Dataset folder used: {RootPath.get_dataset_path()}")

    dataset_volume_path = '/home/ubuntu/new'

    # TBD: correct the path with the real one!
    parquet_dataset_path = os.path.join(dataset_volume_path,"train")
    parquet_dataset_Test_path= os.path.join(dataset_volume_path,"test")
    print('Start reading \n')
    cols = [
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
        'engager_hashtag_count_all_avg',
        'engager_hashtag_count_all_max',
        'engager_hashtag_count_reply_avg',
        'engager_hashtag_count_reply_max',
        'engager_hashtag_count_retweet_avg',
        'engager_hashtag_count_retweet_max',
        'engager_hashtag_count_comment_avg',
        'engager_hashtag_count_comment_max',
        'engager_hashtag_count_like_avg',
        'engager_hashtag_count_like_max',
        'engager_language_count_all',
        'engager_language_count_reply',
        'engager_language_count_retweet',
        'engager_language_count_comment',
        'engager_language_count_like',
        'engager_photo_count_all',
        'engager_photo_count_reply',
        'engager_photo_count_retweet',
        'engager_photo_count_comment',
        'engager_photo_count_like',
        'engager_gif_count_all',
        'engager_gif_count_reply',
        'engager_gif_count_retweet',
        'engager_gif_count_comment',
        'engager_gif_count_like',
        'engager_video_count_all',
        'engager_video_count_reply',
        'engager_video_count_retweet',
        'engager_video_count_comment',
        'engager_video_count_like',
        'engager_type_count_all',
        'engager_type_count_reply',
        'engager_type_count_retweet',
        'engager_type_count_comment',
        'engager_type_count_like',
        'language_hourbin_count_all',
        'language_hourbin_count_reply',
        'language_hourbin_count_retweet',
        'language_hourbin_count_comment',
        'language_hourbin_count_like',
        'engager_weekday_count_all',
        'engager_weekday_count_reply',
        'engager_weekday_count_retweet',
        'engager_weekday_count_comment',
        'engager_weekday_count_like'

    ]

    df = dd.read_parquet(parquet_dataset_path, engine='fastparquet', columns=cols)

    dfTest = dd.read_parquet(parquet_dataset_Test_path, engine='fastparquet', columns=cols)

    df = df.sample(frac = 1)
    

    df.columns=[c.replace("'",'').replace(",","") for c in df.columns]
    dfTest.columns=[c.replace("'",'').replace(",","") for c in dfTest.columns]
    

    # Drop other engagements
    chosen=engCols[idx]
    rest=[c for c in engCols if c!=chosen]

    # Drop other engagements
    df = df.drop(columns=rest)

    df[chosen] = df[chosen].mask(df[chosen] < 0, 0)
    df[chosen] = df[chosen].mask(df[chosen] > 0, 1)

    dfTest = dfTest.drop(columns=rest)

    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] < 0, 0)
    dfTest[chosen] = dfTest[chosen].mask(dfTest[chosen] > 0, 1)
    
    print('Start y \n')
    yTest = dfTest[chosen]
    dfTest = dfTest.drop(columns=[chosen])

    y = df[chosen]
    df = df.drop(columns=[chosen])

    print('Start compute \n')

    # From Dask to Pandas
    df = df.compute()
    y = y.compute()

    print('Start compute \n')

    # From Dask to Pandas
    dfTest = dfTest.compute()
    yTest = yTest.compute()



    print('Columns name:', df.columns)

    X_train, X_val, y_train, y_val = train_test_split(df, y, stratify=y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, stratify=y_val, test_size=0.5)
    gc.collect()
    train = lgb.Dataset(X_train, y_train)
    validation = lgb.Dataset(X_val, y_val)
    gc.collect()
        

    lightGBM_naive = tune_lightGBM_sequential(train, validation)

    with open(f"best_model_{chosen.replace('engagement_','').replace('_timestamp','')}.txt", 'a+') as f:
        f.write(str(lightGBM_naive))


