from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map, count_array_feature
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import pyarrow as pa

dict_name = "domains_id"
out_cols = ["mapped_domains_id", 'tweet_domains_count']
out_frame_name = "mapped_domains"


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map Domains

    # Load dataset
    df = read_dataset(dataset_path, ["raw_feature_tweet_domains"])

    generate_dict, _ = parse_args()

    # Create Dict
    if generate_dict:
        create_and_save_uniques(dict_path, dict_name, df["raw_feature_tweet_domains"], '\t')

    # Map the feature
    out = load_uniques_create_dict_map(c, dict_path, dict_name,
                                       df["raw_feature_tweet_domains"], out_cols[0], np.uint32,
                                       '\t').to_frame()
    out['tweet_domains_count'] = count_array_feature(out['mapped_domains_id'], np.uint8)

    # Write the output dataset
    save_dataset(temp_output_path, out, name='mapped_domains',
                 schema={"mapped_domains_id": pa.list_(pa.uint32()), 'tweet_domains_count': pa.uint8()})
