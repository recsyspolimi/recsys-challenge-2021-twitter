import os
import time
import gc
import json, ujson
import pandas as pd
import dask.dataframe as dd
import dask
from Scripts.Feature_extraction.target_encoder import MTE_one_shot
from Scripts.utilities import save_dataset, read_dataset, parse_args
from Scripts.utilities import start_correct_cluster
from Scripts.Feature_extraction.feature_extraction_utilities import dataset_path, dict_path, temp_output_path, output_path


def load_all_dics(paths):
    all_dics = {}
    for p in paths:
        with open(p, 'r') as f:
            dic = ujson.load(f)
        all_dics[p] = dic
    return all_dics

def addMeta(meta, paths):
    for p in paths:
        out_col = os.path.basename(p)
        out_col.replace('[', '(').replace(']', ')')  # For xgboost
        meta[out_col] = 'float32'
    return meta

def apply_TE_on_test(df, paths):
    # Inputs: df,    the test dataframe.
    #         paths, the list of all TE dictionary paths.
    for dic_path in paths:
        #mean = dic['mean']
        #dic.pop('mean')
        out_col = os.path.basename(dic_path)
        if ('(' in out_col):
            cols = out_col.replace('TE_', '').replace('_engagement_like', '').replace('_engagement_reply', '').replace('_engagement_retweet', '').replace('_engagement_comment', '')
            cols = json.loads(cols.replace("'", '"').replace("(", '[').replace(")", ']'))
        else:
            cols = [out_col.replace('TE_', '').replace('_engagement_like', '').replace('_engagement_reply', '').replace('_engagement_retweet', '').replace('_engagement_comment', '')]
        out_col.replace('[', '(').replace(']', ')')  # For xgboost
        print('Mapping feature TE dict: ', out_col)
        print('Cols:', cols)

        df[out_col] = df[cols[0]].astype('str')
        if (len(cols) > 1):
            for i in range(1, len(cols)):
                df[out_col] = df[out_col] + '_' + df[cols[i]].astype('str')

        dic = {}
        # Slower
        #dic = parse_json_features(dic_path, dic)
        # Faster, may use more ram
        # Fastest: install package ujson
        with open(dic_path, 'r') as f:
            dic = ujson.load(f)
        df[out_col] = df[out_col].map(lambda x: dic[x] if x in dic else float(dic['$mean']))
        #del dic
        #gc.collect()
    return df

def apply_TE_on_test_fast(df, dics):
    # Inputs: df,    the test dataframe.
    #         paths, the list of all TE dictionary paths.
    for p, dic in dics.items():
        #mean = dic['mean']
        #dic.pop('mean')
        out_col = os.path.basename(p)
        if ('(' in out_col):
            cols = out_col.replace('TE_', '').replace('_engagement_like', '').replace('_engagement_reply', '').replace('_engagement_retweet', '').replace('_engagement_comment', '')
            cols = json.loads(cols.replace("'", '"').replace("(", '[').replace(")", ']'))
        else:
            cols = [out_col.replace('TE_', '').replace('_engagement_like', '').replace('_engagement_reply', '').replace('_engagement_retweet', '').replace('_engagement_comment', '')]
        out_col.replace('[', '(').replace(']', ')')  # For xgboost
        print('Mapping feature TE dict: ', out_col)
        print('Cols:', cols)

        df[out_col] = df[cols[0]].astype('str')
        if (len(cols) > 1):
            for i in range(1, len(cols)):
                df[out_col] = df[out_col] + '_' + df[cols[i]].astype('str')

        df[out_col] = df[out_col].map(lambda x: dic[x] if x in dic else float(dic['$mean']))
    return df

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    output_df_path = output_path
    valid_dir = temp_output_path + '/Split_cols/'
    df = read_dataset(valid_dir, None)
    if 'raw_feature_tweet_text_token' in df.columns:
        df = df.drop('raw_feature_tweet_text_token', axis=1)

    dic_dir = dict_path
    dic_paths = [os.path.join(dic_dir, f) for f in os.listdir(dic_dir) if 'TE' in f]

    if is_test:
        # Load one dictionary at time, to avoid OOM
        df = df.compute()
        df = apply_TE_on_test(df, dic_paths)
        d = output_df_path + '/Test_with_TE/'
        if not os.path.exists(d):
            os.makedirs(d)
        #f = os.path.join(d, 'part.0.parquet')
        f = os.path.join(d, 'part.0.parquet')
        df.to_parquet(f)
    else:
        # Load all dictionaries at once, faster for local validation
        print('Loading all dictionaries...')
        all_dics = load_all_dics(dic_paths)
        print('Dictionaries loaded!')

        v_paths = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if 'parquet' in f]
        n_chunks_one_shot = 1
        range_paths = dict(zip(range(1, len(v_paths) + 1), v_paths))
        i = 1
        tmp_paths = []
        while i <= len(v_paths):
            tmp_paths.append(range_paths[i])
            if i % n_chunks_one_shot == 0 or i == len(v_paths):
                print('Computing chunk ', str(i))
                start_time = time.time()
                valid = dd.read_parquet(tmp_paths)
                valid = valid.compute()
                valid = apply_TE_on_test_fast(valid, all_dics)
                # now = datetime.now().strftime('%b%d_%H-%M-%S')
                d = output_df_path + '/Valid_with_TE/'
                if not os.path.exists(d):
                    os.makedirs(d)
                f = os.path.join(d, 'part.' + str(i) + '.parquet')
                valid.to_parquet(f)
                tmp_paths = []
                del valid
                gc.collect()
                print('Chunk time ', time.time() - start_time)
                print('#'*10)
            i = i + 1

    ''' print('Compute started...')
    df = df.compute()
    print('Compute done!')
    df = apply_TE_on_test_fast(df, all_dics)

    df = dd.from_pandas(df, npartitions=250)
    save_dataset(output_df_path, df, "Valid_with_TE")'''


