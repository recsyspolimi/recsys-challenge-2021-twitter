#!/env/bin/python

from catboost import CatBoost, Pool, CatBoostClassifier
import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

def getFirst(n):
    return n[0]

def getFirstValuePrediction(pred):
    return np.array(list(map(getFirst, pred)))

                  ''' columns = [ 
                        'mapped_language_id', 
                        'mapped_tweet_type',
                    ] '''

def evaluate_test_set():
    part_files = [os.path.join('./test', f) for f in os.listdir('./test') if 'part' in f]
    df_from_each_file = (pd.read_csv(f, sep='\x01', header=None) for f in part_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.columns = ['text_tokens', 'hashtags', 'tweet_id', 'media', 'links','domains', 'tweet_type',     
                  'language', 'tweet_timestamp', 'a_user_id', 'creator_follower_count','creator_following_count', 'creator_is_verified',          
                  'creator_creation_timestamp', 'engaging_user_id','engager_follower_count', 'engager_following_count', 'engager_is_verified',         
                  'engager_creation_timestamp', 'engagement_creator_follows_engager']

    base = df[['tweet_id', 'engaging_user_id']]

    df['number_of_photo'] = df['media'].apply(lambda x: str(x).count('Photo') if not(pd.isnull(x)) else 0)
    df['number_of_gif'] = df['media'].apply(lambda x: str(x).count('GIF') if not(pd.isnull(x)) else 0)
    df['number_of_video'] = df['media'].apply(lambda x: str(x).count('Video') if not(pd.isnull(x)) else 0)

    df.drop(['text_tokens', 'hashtags', 'tweet_id', 'media', 'links', 'domains', 'tweet_type', 'language',
             'a_user_id', 'engaging_user_id'], axis=1, inplace=True)

    like_classifier = CatBoostClassifier()
    reply_classifier = CatBoostClassifier()
    retweet_classifier = CatBoostClassifier()
    quote_classifier = CatBoostClassifier()

    like_classifier.load_model("like_classifier", format='cbm')
    reply_classifier.load_model("reply_classifier", format='cbm')
    retweet_classifier.load_model("retweet_classifier", format='cbm')
    quote_classifier.load_model("comment_classifier", format='cbm')
    
    like_prediction = like_classifier.predict_proba(df)
    reply_prediction = reply_classifier.predict_proba(df)
    retweet_prediction = retweet_classifier.predict_proba(df)
    quote_prediction = quote_classifier.predict_proba(df)

    base['reply_pred'] = getFirstValuePrediction(reply_prediction)
    base['retweet_pred'] = getFirstValuePrediction(retweet_prediction)
    base['quote_pred'] = getFirstValuePrediction(quote_prediction)
    base['fav_pred'] = getFirstValuePrediction(like_prediction)
    
    del df
    
    base.to_csv('results.csv', index=False, header=False)

if __name__ == "__main__":
    evaluate_test_set()