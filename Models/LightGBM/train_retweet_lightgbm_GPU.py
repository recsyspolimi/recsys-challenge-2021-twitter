from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import os, sys
import time
import lightgbm as lgb
import RootPath
from Scripts.utilities import start_cluster



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

    if RootPath.is_aws():
        print("Detected running on AWS!")
        c = start_cluster(n_workers=1, threads_per_worker=32, memory_limit="200GB", processes=False)

    else:
        print("Running on local")
    print(f"Dataset folder used: {RootPath.get_dataset_path()}")

    # TBD: correct the path with the real one!
    parquet_dataset_path = os.path.join(RootPath.get_dataset_path(), 'All_feature_dataset')

    print('Start reading \n')


    df = dd.read_parquet(parquet_dataset_path, engine='fastparquet')

    print('Start dropping \n')

    print('Columns name:', df.columns)
    print('Columns type:', df.dtypes)

# Drop useless columns
    df = df.drop(columns=['tweet_timestamp', 'creator_creation_timestamp', 'engager_creation_timestamp',
                          'mapped_creator_id', 'mapped_engager_id',
                          'mapped_tweet_links_id', 'mapped_domains_id', 'mapped_tweet_hashtags_id',
                          'mapped_tweet_id', 'engager_hashtag_count_list',  'tweet_datetime', 'engager_creation_datetime', 'creator_creation_datetime'])

    # Drop other engagements
    df = df.drop(columns=['engagement_reply_timestamp', 'engagement_comment_timestamp', 'engagement_like_timestamp'])

    # Maybe there is a smarter way? It's 01:48, I will leave these 2 lines as they are for the moment...
    df['engagement_retweet_timestamp'] = df['engagement_retweet_timestamp'].mask(df['engagement_retweet_timestamp'] < 0, 0)
    df['engagement_retweet_timestamp'] = df['engagement_retweet_timestamp'].mask(df['engagement_retweet_timestamp'] > 0, 1)

    print('Start y \n')
    y = df['engagement_retweet_timestamp']
    df = df.drop(columns=['engagement_retweet_timestamp'])

    print('Start compute \n')
    # From Dask to Pandas
    df = df.compute()
    y = y.compute()

    print('Columns name:', df.columns)

    X_train, X_val, y_train, y_val = train_test_split(df, y, stratify=y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, stratify=y_val, test_size=0.5)
    del df, y

    train = lgb.Dataset(X_train, y_train)
    validation = lgb.Dataset(X_val, y_val)

    params = {
        'learning_rate': 0.2,
        'objective': 'binary',
        'metric': 'binary',
        'num_threads': -1,
        'num_iterations': 15000,
        'num_leaves': 31,
        'max_depth': 14,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'max_bin': 255,
        'early_stopping_rounds': 20,
        'device' : 'gpu'
    }

    model = lgb.train(params,
                      train_set=train,
                      valid_sets=validation,
                      valid_names='validation')

    model.save_model(f'lightgbm_retweet_{time.time()}.txt', num_iteration=model.best_iteration)

    preds = model.predict(X_test)

    print(f'The model scored a RCE of: {rce(y_test, preds)}')
    print(f'The model scored an AP of: {ap(y_test, preds)}')
