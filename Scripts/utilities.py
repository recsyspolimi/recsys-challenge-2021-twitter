from typing import List, Optional

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import os, time

import RootPath


def read_dataset(dataset_path: str, columns: Optional[List[str]]) -> dd.DataFrame:
    if columns:
        #print(f"Loading {columns=}")
        return dd.read_parquet(dataset_path,
                               columns=columns,
                               engine='fastparquet'
                               #engine='pyarrow'
                               )
    else:
        print(f"Loading all columns")
        return dd.read_parquet(dataset_path,
                               engine='fastparquet',
                               #engine='pyarrow'
                               )


def save_dataset(temp_output_path: str, data: dd.DataFrame, name: Optional[str] = None, schema=None):
    if not name:
        assert len(data.columns) == 1
        name = data.columns[0]

    print(f"Saving output dataset: {name}")
    start = time.time()
    data.to_parquet(os.path.join(temp_output_path, name),
                    write_index=False,
                    compression="snappy",
                    engine="pyarrow",
                    #engine='fastparquet',
                    overwrite="True",
                    schema=schema
                    )
    end = time.time()
    print(f"Done. Time elapsed: {end-start}")


# def save_dict(dict_path:str, name:str, client: Client):
#     def helper(name):
#         w = get_worker()
#         dictionary = w.direct_dict
#
#         print(f"Saving mapping dictionary: {name=}, with size {sys.getsizeof(dictionary)/(10**6)} MB")
#         start = time.time()
#         print("dictionary type:", type(dictionary))
#         with gzip.GzipFile(os.path.join(dict_path, name + "_dict"), 'wb') as file:
#             pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
#         end = time.time()
#         print(f"Done. Time elapsed: {end-start}")
#
#     result = client.submit(helper, name)
#     result.result()


def start_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB", processes=True):
    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit, processes=processes
    )
    client = Client(cluster)  # use default n_threads and mem
    print(client)
    print(client.cluster)
    return client


def start_correct_cluster(is_test, use_processes):
    if is_test:
        c = start_cluster(n_workers=1, threads_per_worker=1, memory_limit="64GB", processes=False)
    elif RootPath.is_aws():
        if use_processes:
            c = start_cluster(n_workers=8, threads_per_worker=1, memory_limit="32GB", processes=True)
        else:
            c = start_cluster(n_workers=1, threads_per_worker=32, memory_limit="256GB", processes=False)
    else:
        if use_processes:
            c = start_cluster(n_workers=2, threads_per_worker=1, memory_limit="4GB", processes=True)
        else:
            c = start_cluster(n_workers=1, threads_per_worker=4, memory_limit="6GiB", processes=False)
    return c


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_dict_generation', action='store_false', dest='generate_dict')
    parser.add_argument('--is_test', action='store_true', dest='is_test')

    args = parser.parse_args()

    return args.generate_dict, args.is_test
