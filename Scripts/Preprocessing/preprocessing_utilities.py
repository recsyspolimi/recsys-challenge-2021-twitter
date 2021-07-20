import gzip
from typing import Optional

from dask.distributed import Client, get_worker
import dask
import dask.dataframe as dd
import RootPath
import pandas as pd
import numpy as np
import time
import os
import json

with open(os.path.join(RootPath.get_root(), "Scripts", "Preprocessing", "paths.json"), "r") as read_file:
    config = json.load(read_file)

original_dataset = config['original_dataset']
base_path = config['base_path']
temp_path = config['temp_path']
dataset_path = os.path.join(temp_path, 'Full_dataset_parquet')
output_path = os.path.join(base_path, 'Full_mapped_dataset')
temp_output_path = os.path.join(temp_path, 'Columns')
dict_path = config['dict_path']
train_val_ratio = config['train_val_ratio']


def create_and_save_uniques(path_to_uniques_dir: str, name: str, series: dask.dataframe.Series, sep: Optional[str] = None) -> None:
    """
    Extracts unique values from dask series, and stores it as pickled numpy array in the path specified.

    Args:
        path_to_uniques_dir: path to the directory in which all unique values of all series are stored
        name: name of the saved file containing the unique values
        series: the dask Series containing all the data from which extract the unique values
        sep: optional argument, the separator to use in case of splitting of each line needed to extract uniques

    Returns:
        None
    """
    s_name = series.name
    start = time.time()
    if not sep:
        print(f"Creating uniques of feature: {s_name}")
        uniques = series.drop_duplicates().astype('S32')
    else:
        print(f"Creating uniques of feature to split: {s_name}")
        uniques = series.map_partitions(
            lambda s: pd.Series(np.array([elem for line in s if line != ""
                                               for elem in line.split(sep)], dtype='S32'), name=series.name),
            meta=pd.Series(dtype=str, name=series.name)
        ).drop_duplicates()

    uniques, = dask.compute(uniques)

    arr = uniques.values
    print(f"Got uniques array. Infos:")
    info = {
        "len": len(arr),
        "memory": arr.nbytes / (10 ** 6),
        "dtype": arr.dtype.name
    }
    print(json.dumps(info, indent=4, separators=(',', ': ')))

    path = os.path.join(path_to_uniques_dir, name)
    print(f"Saving array in {path}")

    # uniques.to_pickle(path, compression='gzip')
    with gzip.GzipFile(path, 'wb') as file:
        np.save(file, arr)  # best compression-speed tradeoff with this technique
    #     pickle.dump(uniques, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path_to_uniques_dir, name + ".json"), "w+") as write_file:
        json.dump(info, write_file, indent=4, separators=(',', ': '))

    end = time.time()
    print(f"Done. Time elapsed: {end - start}")
    print(f"Some examples of detected uniques:")
    for i in range(min(5, len(arr))):
        print(arr[i], type(arr[i]))
    return None


def load_uniques_create_dict_map(client: dask.distributed.Client, path_to_uniques_dir: str, name_unique: str,
                                 series: dask.dataframe.Series, name_out: str, out_type: type,
                                 sep:Optional[str] = None, nan_symbol: str = "") -> dask.dataframe.Series:
    """
    Loads a numpy array from which it builds the dictionary to map over a feature. It then maps it on the input series

    Args:
        client: the client used in the computation, to which load the dictionary
        path_to_uniques_dir:  path to the directory used to retrieve the unique values array
        name_unique: the name of the file containing the needed numpy array
        series: the series to be mapped using built dictionary
        name_out: the name of the output mapped series
        out_type: the type to be imposed onto the output mapped series
        sep: optional argument, the separator to use in case of splitting of each line needed to map values
        nan_symbol: the Nan symbol to be ignored and mapped to an empty list in case of splitting features

    Returns:
        the output mapped series
    """

    def load_dict(path_to_uniques):
        # uniques = pd.read_pickle(path_to_uniques, "gzip")
        with gzip.GzipFile(path_to_uniques, 'rb') as file:
            arr = np.load(file, allow_pickle=True)
        #     uniques = pickle.load(file)
        uniques = pd.Series(arr)
        # required to have a smaller final dict, since by doing this final array will be of class 'bytes'
        # and not of class'np.bytes' (larger)

        info = {
            "len": len(arr),
            "memory": arr.nbytes / (10 ** 6),
            "dtype": arr.dtype.name
        }
        print("Info from loaded uniques")
        print(json.dumps(info))

        d = dict(zip(uniques, range(1, len(uniques) + 1)))
        print("Created dict content:")
        i = iter(d.keys())
        for _ in range(min(5, len(arr))):
            k = next(i)
            print(k, d[k], type(k), type(d[k]))

        w = get_worker()
        w.direct_dict = d
        return "Ended"

    path = os.path.join(path_to_uniques_dir, name_unique)
    print(f"Loading Dict: {path}")
    start = time.time()
    g = client.submit(load_dict, path).result()
    end = time.time()
    print(f"{g} the load+dict build phase! Time needed: {end - start}")

    def use_dict(x):
        x = bytes(x, 'utf-8')
        w = get_worker()
        return w.direct_dict.get(x, 0)  # return 0 if x not in array

    def use_dict_to_split(x):
        w = get_worker()
        return np.array([w.direct_dict.get(bytes(y, 'utf-8'), 0) for y in x.split(sep)], dtype=out_type) if x != nan_symbol else np.array([])
        #return [out_type(w.direct_dict.get(bytes(y, 'utf-8'), 0)) for y in x.split(sep)] if x != nan_symbol else []


    # if x is not pd.NA else None, #Nans to be managed outside, manual entry in dict if you like
    if not sep:
        print(f"Mapping feature: {series.name} using dict")
        return series.map(use_dict, meta=pd.Series(dtype=int, name=series.name)).astype(out_type).rename(name_out)
    else:
        print(f"Mapping feature to split: {series.name} using dict")
        return series.map(use_dict_to_split, meta=pd.Series(dtype='O', name=series.name)).rename(name_out)


def count_array_feature(series: dask.dataframe.Series, out_type: type,) -> dask.dataframe.Series:
    count_f = np.vectorize(len)

    # def count_series(s):
    #     # count_s = count_f(s.values)
    #     # return pd.Series(count_s, dtype=out_type)
    #     return count_f(s).astype(out_type)

    res = series.map_partitions(count_f, meta=('', np.int_)).astype(out_type)
    return res


def count_unique_array_feature(series: dask.dataframe.Series, out_type: type,) -> dask.dataframe.Series:
    count_f_unique = np.vectorize(lambda arr: len(np.unique(arr)))
    count_f = np.vectorize(len)

    # def count_series(s):
    #     count_s = count_f(s.values)
    #     return pd.Series(count_s, dtype=out_type)

    def custom(arr):
        l = len(arr)
        if l <= 1:
            return l
        ret = 0
        np.ndarray.sort(arr) #inplace sort! No allocation of memory for intermediate results!
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                ret += 1
            elif arr[i] < arr[i - 1]:
                assert False
        return ret


    #res = series.map_partitions(, meta=('', 'O')).map_partitions(count_f, meta=('', np.int_)).astype(out_type)
    #res = series.map(custom, meta=('', np.int_)).astype(out_type)
    #res = series.map_partitions(count_f_unique, meta=("", np.int_))
    #res = series.map(lambda arr: len(np.unique(arr)))
    res = series.map(custom, meta=("", np.int_))  # faster thanks to not allocating memory
    return res

# OLD LEGACY CODE BELOW

# def create_dict_feature(series: dask.dataframe.Series, client: dask.distributed.Client, split_every=8, split_out=1) -> dask.distributed.Future:
#
#     def helper(series, split_every, split_out):
#         feature_name = series.name
#         feature_name_encode = feature_name + "_encode"
#         print(f"Creating dict of feature: {feature_name=}")
#
#         start = time.time()
#         #mapping = series.map_partitions(lambda d: d.drop_duplicates()).drop_duplicates(split_every=split_every, split_out=split_out).to_frame()  # create a dataframe from a series
#         mapping = series.drop_duplicates().to_frame()  # create a dataframe from a series
#         mapping[feature_name_encode] = 1
#         mapping[feature_name_encode] = mapping[feature_name_encode].cumsum()
#
#         #mapping, = dask.compute(mapping)
#         mapping, = dask.persist(mapping)
#         print(mapping.head(10))
#
#         print("---- creating dict ---------!")
#
#         #direct_dict = dict(zip(mapping[feature_name], mapping[feature_name_encode]))
#
#         w = get_worker()
#         w.direct_dict = {}
#
#         def save_one_partition(df):
#             new_dict = dict(zip(df[feature_name], df[feature_name_encode]))
#             w = get_worker()
#             w.direct_dict.update(new_dict)
#
#             return None
#
#         res = mapping.map_partitions(save_one_partition)
#         res, = dask.compute(res)
#         print(len(res), res)
#
#         end = time.time()
#         print("Max id: {}".format(len(w.direct_dict)))
#         print(f"Done. Time elapsed: {end - start}")
#         print(f"Dict size: {sys.getsizeof(w.direct_dict)/(10**6)=} MB")
#
#         return w.direct_dict
#
#     direct_dict = client.submit(helper, series, split_every, split_out)
#     direct_dict.result()
#
#
# def create_dict_feature_to_split(series: dask.dataframe.Series, sep: str, client: dask.distributed.Client, split_every=8, split_out=1) -> dask.distributed.Future:
#
#     def helper(series, sep, split_every, split_out):
#         feature_name = series.name
#         feature_name_encode = feature_name + "_encode"
#
#         print(f"Creating dict of feature: {feature_name=}")
#         start = time.time()
#         # map partition internal function goes from series to dataframe
#         mapping = series \
#             .map_partitions(
#             lambda s: pd.DataFrame([hashtag for line in s.dropna() for hashtag in line.split(sep)], columns=[feature_name]),
#             meta={feature_name: pd.StringDtype()}) \
#             .drop_duplicates(split_every=split_every, split_out=split_out)
#         mapping[feature_name_encode] = 1
#         mapping[feature_name_encode] = mapping[feature_name_encode].cumsum()
#
#         # mapping, = dask.compute(mapping)
#         mapping, = dask.persist(mapping)
#
#         print("---- creating dict ---------!")
#
#         # direct_dict = dict(zip(mapping[feature_name], mapping[feature_name_encode]))
#
#         w = get_worker()
#         w.direct_dict = {}
#
#         def save_one_partition(df):
#             new_dict = dict(zip(df[feature_name], df[feature_name_encode]))
#             w = get_worker()
#             w.direct_dict.update(new_dict)
#
#             return None
#
#         res = mapping.map_partitions(save_one_partition).compute()
#         print(len(res), res)
#
#         end = time.time()
#         print(f"Done. Time elapsed: {end - start}")
#         print(f"Dict size: {sys.getsizeof(w.direct_dict)/(10**6)=} MB")
#
#         return w.direct_dict
#
#     direct_dict = client.submit(helper, series, sep, split_every, split_out)
#
#     return direct_dict.result()
#
#
# def map_column_single_value(series: dask.dataframe.Series, name_out: str,
#                             out_type: type) -> dask.dataframe.Series:
#     def use_dict(x):
#         w = get_worker()
#         return w.direct_dict[x]
#
#     # if x is not pd.NA else None, #Nans to be managed outside, manual entry in dict if you like
#     return series.map(use_dict, meta=pd.Series(dtype=int, name=series.name)).astype(out_type).rename(name_out)
#
#
# def map_column_array(series: dask.dataframe.Series, sep: str, name_out: str, out_type: type, nan_symbol) -> dask.dataframe.Series:
#     def use_dict(x):
#         w = get_worker()
#         return np.array([w.direct_dict[y] for y in x.split(sep)], dtype=out_type) if x is not nan_symbol else np.array([])
#
#     return series.map(use_dict, meta=pd.Series(dtype='O', name=series.name)).rename(name_out)
#
#
# def factorize_create_mapping(series: dd.Series, out_type: type) -> dd.DataFrame:
#     feature_name = series.name
#     feature_name_encode = feature_name + "_encode"
#
#     mapping = series.drop_duplicates().to_frame()  # create a dataframe from a series
#     mapping[feature_name_encode] = 1
#     mapping[feature_name_encode] = mapping[feature_name_encode].cumsum()
#     mapping[feature_name_encode] = mapping[feature_name_encode].astype(out_type)
#     mapping, = dask.persist(mapping)
#     return mapping
#
#
# def factorize_do_merge(df: dd.DataFrame, mapping: dd.DataFrame, left_on=None, right_on=None, left_index=False, right_index=False) -> dd.DataFrame:
#     feature_name_encode = right_on + "_encode"
#
#     df = df.merge(mapping, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, how='left')
#     df = df.drop([left_on, right_on], axis=1)
#     df.columns = [i if i != feature_name_encode else left_on for i in df.columns]
#     return df