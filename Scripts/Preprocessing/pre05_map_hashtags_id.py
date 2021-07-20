from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map, count_array_feature, count_unique_array_feature
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import pyarrow as pa

dict_name = "hashtags_id"
out_cols = ["mapped_tweet_hashtags_id", 'tweet_hashtags_count', 'tweet_hashtags_unique_count']
out_frame_name = "mapped_hashtags"


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map Hashtags

    # Load dataset
    df = read_dataset(dataset_path, ["raw_feature_tweet_hashtags"])

    # Create Dict
    if generate_dict:
        create_and_save_uniques(dict_path, dict_name, df["raw_feature_tweet_hashtags"], '\t')

    # Map the feature
    out = load_uniques_create_dict_map(c, dict_path, dict_name,
                                       df["raw_feature_tweet_hashtags"], out_cols[0], np.uint32,
                                       '\t').to_frame()
    out['tweet_hashtags_count'] = count_array_feature(out['mapped_tweet_hashtags_id'], np.uint8)
    out['tweet_hashtags_unique_count'] = count_unique_array_feature(out['mapped_tweet_hashtags_id'], np.uint8)

    # Write the output dataset
    save_dataset(temp_output_path, out,
                 name='mapped_hashtags',
                 schema={"mapped_tweet_hashtags_id": pa.list_(pa.uint32()),
                         'tweet_hashtags_count': pa.uint8(),
                         'tweet_hashtags_unique_count': pa.uint8()
                         })
