from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map, count_array_feature
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import RootPath
import pyarrow as pa

dict_name = 'links_id'
out_cols = ['mapped_tweet_links_id', 'tweet_links_count']
out_frame_name = "mapped_links"

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map Links

    # Load dataset
    df = read_dataset(dataset_path, ["raw_feature_tweet_links"])

    # Create Dict
    if generate_dict:
        create_and_save_uniques(dict_path, dict_name, df["raw_feature_tweet_links"], '\t')

    # Map the feature
    out = load_uniques_create_dict_map(c, dict_path, dict_name,
                                       df["raw_feature_tweet_links"], 'mapped_tweet_links_id', np.uint32,
                                       '\t').to_frame()
    out['tweet_links_count'] = count_array_feature(out['mapped_tweet_links_id'], np.uint8)

    # Write the output dataset
    save_dataset(temp_output_path, out,
                 name='mapped_links',
                 schema={"mapped_tweet_links_id": pa.list_(pa.uint32()), 'tweet_links_count': pa.uint8()})
