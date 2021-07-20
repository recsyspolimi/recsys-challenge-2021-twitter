from typing import Optional

from Scripts.utilities import start_correct_cluster, read_dataset, parse_args, save_dataset
from preprocessing_utilities import output_path, base_path, train_val_ratio
import time, os
import dask
import dask.dataframe as dd
import pyarrow.parquet as pq


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False) # better False

    if generate_dict:
        # Load dataset
        df = read_dataset(output_path, columns=None)
        schema = pq.read_schema(os.path.join(output_path, "_metadata"))
        tot_parts = df.npartitions

        official_train_df = df[~df['is_from_official_val']]
        official_val_df = df[df['is_from_official_val']]

        train_train_df, train_val_df = official_train_df.random_split(train_val_ratio, random_state=123)
        val_train_df, val_val_df = official_val_df.random_split([0.7, 0.3], random_state=123)

        train_df = dd.concat([train_train_df, val_train_df], axis=0)
        val_df = dd.concat([train_val_df, val_val_df], axis=0)

        train_df = train_df.repartition(partition_size="200MB")
        val_df = val_df.repartition(partition_size="200MB")

        print(f"Splitting Dataset in Train and Valid. Train_perc: {train_val_ratio[0]}, Valid_perc:{train_val_ratio[1]}")

        print(f"Saving output datasets")
        start = time.time()
        save_dataset(base_path, train_df, 'Train', schema)
        save_dataset(base_path, val_df, 'Valid', schema)
        end = time.time()
        print(f"Done. Time elapsed: {end - start}.")
